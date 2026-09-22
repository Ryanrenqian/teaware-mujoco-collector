from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class IKSolution:
    q: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    converged: bool


def pose_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, np.asarray(quaternion_wxyz, dtype=np.float64))
    matrix[:3, :3] = rotation.reshape(3, 3)
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def matrix_quaternion(matrix: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=np.float64)[:3, :3].reshape(-1))
    return quaternion


def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 0.5 * sum(np.cross(current[:, index], target[:, index]) for index in range(3))


class MuJoCoIKPlanner:
    """Damped-least-squares IK using the exact MuJoCo collection model."""

    def __init__(self, model: mujoco.MjModel, robots: list[dict[str, Any]]) -> None:
        self.model = model
        self.robots = robots
        self.data = mujoco.MjData(model)

    def _set_robot_state(
        self,
        robot_index: int,
        arm_q: np.ndarray,
        hand_q: np.ndarray | None = None,
    ) -> None:
        runtime = self.robots[robot_index]
        self.data.qpos[runtime["arm_qpos_addresses"]] = arm_q
        if hand_q is not None:
            dof = runtime["hand_dof"]
            self.data.qpos[runtime["hand_qpos_addresses"]] = hand_q[:dof]
        mujoco.mj_forward(self.model, self.data)

    def solve_tcp(
        self,
        robot_index: int,
        start_q: np.ndarray,
        target_pose: np.ndarray,
        *,
        hand_q: np.ndarray | None = None,
        max_iterations: int = 240,
        position_tolerance_m: float = 0.006,
        orientation_tolerance_rad: float = 0.08,
        orientation_weight: float = 0.35,
        damping: float = 0.04,
        max_step_rad: float = 0.10,
    ) -> IKSolution:
        runtime = self.robots[robot_index]
        q = np.asarray(start_q, dtype=np.float64).reshape(7).copy()
        target = np.asarray(target_pose, dtype=np.float64).reshape(4, 4)
        joint_ids = runtime["arm_joint_ids"]
        lower = self.model.jnt_range[joint_ids, 0]
        upper = self.model.jnt_range[joint_ids, 1]
        dof_addresses = runtime["arm_dof_addresses"]
        site_id = int(runtime["tcp_site_id"])
        jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
        jac_rot = np.zeros((3, self.model.nv), dtype=np.float64)

        position_error = float("inf")
        orientation_error = float("inf")
        for _ in range(int(max_iterations)):
            self._set_robot_state(robot_index, q, hand_q)
            current_position = self.data.site_xpos[site_id]
            current_rotation = self.data.site_xmat[site_id].reshape(3, 3)
            position_delta = target[:3, 3] - current_position
            rotation_delta = _rotation_error(current_rotation, target[:3, :3])
            position_error = float(np.linalg.norm(position_delta))
            orientation_error = float(np.linalg.norm(rotation_delta))
            if (
                position_error <= position_tolerance_m
                and orientation_error <= orientation_tolerance_rad
            ):
                return IKSolution(q, position_error, orientation_error, True)

            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, site_id)
            jacobian = np.vstack(
                [jac_pos[:, dof_addresses], orientation_weight * jac_rot[:, dof_addresses]]
            )
            error = np.concatenate([position_delta, orientation_weight * rotation_delta])
            normal = jacobian @ jacobian.T + damping**2 * np.eye(6)
            delta_q = jacobian.T @ np.linalg.solve(normal, error)
            delta_q = np.clip(delta_q, -max_step_rad, max_step_rad)
            q = np.clip(q + delta_q, lower, upper)

        return IKSolution(q, position_error, orientation_error, False)

    def solve_hand_from_links(
        self,
        robot_index: int,
        arm_q: np.ndarray,
        start_hand_q: np.ndarray,
        target_links_world: dict[str, np.ndarray],
        *,
        max_iterations: int = 180,
        tolerance_m: float = 0.012,
        damping: float = 0.025,
        max_step_rad: float = 0.10,
    ) -> np.ndarray:
        """Fit xHand joints to TRO link positions without Pyroki or waic_demo4."""
        runtime = self.robots[robot_index]
        hand_dof = int(runtime["hand_dof"])
        q = np.asarray(start_hand_q, dtype=np.float64).reshape(-1)[:hand_dof].copy()
        joint_ids = runtime["hand_joint_ids"]
        lower = self.model.jnt_range[joint_ids, 0]
        upper = self.model.jnt_range[joint_ids, 1]
        dof_addresses = runtime["hand_dof_addresses"]
        robot_id = runtime["id"]

        targets: list[tuple[int, np.ndarray]] = []
        for link_name, transform in target_links_world.items():
            body_name = f"{robot_id}_{link_name}"
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0 and link_name not in {"right_hand_link", "left_hand_link"}:
                target = np.asarray(transform, dtype=np.float64).reshape(4, 4)[:3, 3]
                if np.isfinite(target).all():
                    targets.append((body_id, target))
        if not targets:
            return q

        jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
        for _ in range(int(max_iterations)):
            self._set_robot_state(robot_index, arm_q, q)
            errors: list[np.ndarray] = []
            jacobians: list[np.ndarray] = []
            for body_id, target in targets:
                errors.append(target - self.data.xpos[body_id])
                mujoco.mj_jacBody(self.model, self.data, jac_pos, None, body_id)
                jacobians.append(jac_pos[:, dof_addresses].copy())
            error = np.concatenate(errors)
            if float(np.sqrt(np.mean(error**2))) <= tolerance_m:
                break
            jacobian = np.vstack(jacobians)
            normal = jacobian.T @ jacobian + damping**2 * np.eye(hand_dof)
            delta_q = np.linalg.solve(normal, jacobian.T @ error)
            q = np.clip(q + np.clip(delta_q, -max_step_rad, max_step_rad), lower, upper)
        return q
