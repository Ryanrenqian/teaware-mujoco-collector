from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import httpx
import mujoco
import numpy as np

from ..robots import XHAND_LOWER, XHAND_UPPER
from ..teaware_assets import get_teaware_asset, teaware_asset_dir
from .base import ActionChunk, PolicyObservation
from .ik import MuJoCoIKPlanner, matrix_quaternion, pose_matrix
from .tro_runtime import LocalTRORuntime


class TROBackend(Protocol):
    def infer(
        self, object_points: np.ndarray, environment_points: np.ndarray
    ) -> list[dict[str, Any]]: ...

    def metadata(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _encode_npy(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class HTTPTROBackend:
    def __init__(
        self,
        url: str,
        *,
        timeout_s: float,
        hand_type: str,
        num_candidates: int,
    ) -> None:
        self.url = str(url).rstrip("/")
        self.timeout_s = float(timeout_s)
        self.hand_type = str(hand_type)
        self.num_candidates = int(num_candidates)
        self.client = httpx.Client(timeout=self.timeout_s)

    def infer(
        self, object_points: np.ndarray, environment_points: np.ndarray
    ) -> list[dict[str, Any]]:
        response = self.client.post(
            f"{self.url}/v1/grasp-candidates",
            json={
                "hand_type": self.hand_type,
                "num_candidates": self.num_candidates,
                "object_points_npy_base64": _encode_npy(object_points),
                "environment_points_npy_base64": _encode_npy(environment_points),
            },
        )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates", payload) if isinstance(payload, dict) else payload
        if not isinstance(candidates, list):
            raise TypeError("TRO response must contain a candidates list")
        return [dict(candidate) for candidate in candidates]

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "http",
            "url": self.url,
            "hand_type": self.hand_type,
            "num_candidates": self.num_candidates,
        }

    def close(self) -> None:
        self.client.close()


class CentroidMockTROBackend:
    """Integration-only backend; production collection should use local or HTTP TRO."""

    def __init__(self, *, num_candidates: int = 1) -> None:
        self.num_candidates = int(num_candidates)

    def infer(
        self, object_points: np.ndarray, environment_points: np.ndarray
    ) -> list[dict[str, Any]]:
        del environment_points
        center = np.mean(np.asarray(object_points, dtype=np.float64), axis=0)
        top = float(np.quantile(object_points[:, 2], 0.8))
        return [
            {
                "rank": 0,
                "tcp_position_base": [float(center[0]), float(center[1]), top + 0.055],
                "hand_q": [1.35, -0.55, 0.55, 0.0, 1.25, 1.10, 1.25, 1.10, 1.2, 1.05, 1.15, 1.0],
            }
        ]

    def metadata(self) -> dict[str, Any]:
        return {"backend": "centroid_mock", "num_candidates": self.num_candidates}

    def close(self) -> None:
        return None


@lru_cache(maxsize=64)
def _obj_vertices(path: str) -> np.ndarray:
    vertices: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                values = line.split()
                if len(values) >= 4:
                    vertices.append([float(values[1]), float(values[2]), float(values[3])])
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {path}")
    return np.asarray(vertices, dtype=np.float64)


def _sample_rows(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        raise ValueError("point cloud cannot be empty")
    indices = np.linspace(0, len(points) - 1, int(count), dtype=np.int64)
    return points[indices]


def _base_transform(robot: dict[str, Any]) -> np.ndarray:
    yaw = np.radians(float(robot["base_yaw_deg"]))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    transform[:3, 3] = np.asarray(robot["base_position"], dtype=np.float64)
    return transform


class ScenePointCloudBuilder:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        object_points: int,
        environment_points: int,
    ) -> None:
        self.config = config
        self.object_count = int(object_points)
        self.environment_count = int(environment_points)
        self.local_points: dict[str, np.ndarray] = {}
        for spec in config["objects"]:
            asset_id = spec.get("asset_id")
            if asset_id:
                asset = get_teaware_asset(asset_id)
                self.local_points[spec["name"]] = _obj_vertices(
                    str((teaware_asset_dir() / asset["visual"]).resolve())
                )
            else:
                extents = {
                    "teapot": (0.18, 0.12, 0.10),
                    "teacup": (0.09, 0.09, 0.07),
                    "pitcher": (0.13, 0.10, 0.12),
                    "canister": (0.09, 0.09, 0.14),
                }[spec["preset"]]
                self.local_points[spec["name"]] = self._box_points(extents, 512)

    @staticmethod
    def _box_points(extents: tuple[float, float, float], count: int) -> np.ndarray:
        rng = np.random.default_rng(0)
        half = np.asarray(extents, dtype=np.float64) / 2.0
        points = rng.uniform(-half, half, size=(count, 3))
        faces = rng.integers(0, 3, size=count)
        signs = rng.choice([-1.0, 1.0], size=count)
        points[np.arange(count), faces] = half[faces] * signs
        points[:, 2] += half[2]
        return points

    def build(
        self,
        observation: PolicyObservation,
        *,
        target_object: str,
        robot: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            target_index = observation.object_names.index(target_object)
        except ValueError as exc:
            raise ValueError(f"target object {target_object!r} is not in the scene") from exc
        world_from_base = _base_transform(robot)
        base_from_world = np.linalg.inv(world_from_base)
        clouds: list[np.ndarray] = []
        target_cloud: np.ndarray | None = None
        for index, name in enumerate(observation.object_names):
            local = self.local_points[name]
            object_from_local = pose_matrix(
                observation.object_position[index], observation.object_quaternion[index]
            )
            points_world = (object_from_local[:3, :3] @ local.T).T + object_from_local[:3, 3]
            points_base = (base_from_world[:3, :3] @ points_world.T).T + base_from_world[:3, 3]
            if index == target_index:
                target_cloud = points_base
            else:
                clouds.append(points_base)

        table = self.config["table"]
        size = np.asarray(table["size"], dtype=np.float64)
        center = np.asarray(table["center"], dtype=np.float64)
        side = max(4, int(np.sqrt(self.environment_count // 2)))
        x_values = np.linspace(center[0] - size[0] / 2, center[0] + size[0] / 2, side)
        y_values = np.linspace(center[1] - size[1] / 2, center[1] + size[1] / 2, side)
        grid_x, grid_y = np.meshgrid(x_values, y_values)
        table_world = np.column_stack(
            [grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, center[2] + size[2] / 2)]
        )
        table_base = (base_from_world[:3, :3] @ table_world.T).T + base_from_world[:3, 3]
        clouds.append(table_base)
        assert target_cloud is not None
        return (
            _sample_rows(target_cloud, self.object_count).astype(np.float32),
            _sample_rows(np.concatenate(clouds), self.environment_count).astype(np.float32),
        )


def _as_pose(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        return pose_matrix(
            np.asarray(value["position"], dtype=np.float64),
            np.asarray(value["quaternion_wxyz"], dtype=np.float64),
        )
    array = np.asarray(value, dtype=np.float64)
    if array.size == 16:
        return array.reshape(4, 4)
    raise ValueError("pose must be a 4x4 matrix or position/quaternion mapping")


def _smooth_segment(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    alpha = np.linspace(0.0, 1.0, max(2, int(count)), dtype=np.float64)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return start[None] * (1.0 - alpha[:, None]) + end[None] * alpha[:, None]


class TROGraspPolicy:
    def __init__(
        self,
        config: dict[str, Any],
        model: mujoco.MjModel,
        robots: list[dict[str, Any]],
        *,
        backend: TROBackend | None = None,
    ) -> None:
        policy = config["policy"]
        tro = policy["tro"]
        self.config = config
        self.robots = robots
        self.control_hz = float(policy["control_hz"])
        self.action_horizon = int(policy["action_horizon"])
        self.dt_s = 1.0 / self.control_hz
        self.robot_index = next(
            index for index, robot in enumerate(robots) if robot["id"] == policy["robot_id"]
        )
        self.robot = robots[self.robot_index]
        if self.robot["hand"] != "xhand":
            raise ValueError("tro_grasp currently requires an xhand robot")
        self.target_object = str(policy["target_object"])
        self.tro = tro
        self.backend = backend or self._build_backend(tro)
        self.pointclouds = ScenePointCloudBuilder(
            config,
            object_points=tro["object_points"],
            environment_points=tro["environment_points"],
        )
        self.ik = MuJoCoIKPlanner(model, robots)
        self.arm_plan: np.ndarray | None = None
        self.hand_plan: np.ndarray | None = None
        self.stages: list[str] = []
        self.plan_info: dict[str, Any] = {}

    @staticmethod
    def _build_backend(tro: dict[str, Any]) -> TROBackend:
        backend = tro["backend"]
        if backend == "local":
            return LocalTRORuntime(
                root=tro["root"],
                config=tro["config"],
                checkpoint=tro["checkpoint"],
                hand_type=tro["hand_type"],
                device=tro.get("device"),
                num_candidates=tro["num_candidates"],
                inference_steps=tro.get("inference_steps"),
                noise_lambda=tro.get("noise_lambda"),
                root_link_name=tro.get("root_link_name"),
            )
        if backend == "http":
            return HTTPTROBackend(
                tro["url"],
                timeout_s=tro["timeout_s"],
                hand_type=tro["hand_type"],
                num_candidates=tro["num_candidates"],
            )
        if backend == "centroid_mock":
            return CentroidMockTROBackend(num_candidates=tro["num_candidates"])
        raise ValueError(f"unsupported TRO backend: {backend}")

    def _candidate_tcp_world(
        self,
        candidate: dict[str, Any],
        observation: PolicyObservation,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        world_from_base = _base_transform(self.robot)
        predicted_links_world: dict[str, np.ndarray] = {}
        for name, value in dict(candidate.get("pred_links", {})).items():
            predicted_links_world[name] = world_from_base @ _as_pose(value)

        if "tcp_pose_world" in candidate:
            return _as_pose(candidate["tcp_pose_world"]), predicted_links_world
        if "tcp_pose_base" in candidate:
            return world_from_base @ _as_pose(candidate["tcp_pose_base"]), predicted_links_world
        if "tcp_position_base" in candidate:
            target = pose_matrix(
                observation.tcp_position[self.robot_index],
                observation.tcp_quaternion[self.robot_index],
            )
            target[:3, 3] = (
                world_from_base @ np.array([*candidate["tcp_position_base"], 1.0], dtype=np.float64)
            )[:3]
            return target, predicted_links_world

        root_value = candidate.get("root_pose_base", candidate.get("palm_pose_base"))
        if root_value is None:
            raise ValueError("TRO candidate must provide root_pose_base or tcp_pose_base")
        root_world = world_from_base @ _as_pose(root_value)
        root_to_tcp = np.eye(4, dtype=np.float64)
        root_to_tcp[:3, 3] = np.asarray(self.tro["root_to_tcp_position"], dtype=np.float64)
        root_to_tcp[:3, :3] = pose_matrix(
            np.zeros(3), np.asarray(self.tro["root_to_tcp_quaternion_wxyz"], dtype=np.float64)
        )[:3, :3]
        return root_world @ root_to_tcp, predicted_links_world

    def _candidate_hand_q(
        self,
        candidate: dict[str, Any],
        final_arm_q: np.ndarray,
        open_q: np.ndarray,
        predicted_links_world: dict[str, np.ndarray],
    ) -> np.ndarray:
        if "hand_q" in candidate:
            q = np.asarray(candidate["hand_q"], dtype=np.float64).reshape(12)
        else:
            named = candidate.get("finger_q", candidate.get("q_pk"))
            if isinstance(named, dict):
                q = open_q.copy()
                for index, full_name in enumerate(self.robot["hand_joint_names"]):
                    suffix = full_name.removeprefix(f"{self.robot['id']}_")
                    for key in (
                        full_name,
                        suffix,
                        suffix.removeprefix("right_hand_"),
                        suffix.removeprefix("left_hand_"),
                    ):
                        if key in named:
                            q[index] = float(named[key])
                            break
            elif predicted_links_world:
                q = self.ik.solve_hand_from_links(
                    self.robot_index,
                    final_arm_q,
                    open_q,
                    predicted_links_world,
                )
            else:
                q = np.asarray(self.tro["fallback_hand_q"], dtype=np.float64)
        return np.clip(q, np.asarray(XHAND_LOWER), np.asarray(XHAND_UPPER))

    def _plan_candidate(
        self,
        candidate: dict[str, Any],
        observation: PolicyObservation,
    ) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]] | None:
        final_tcp, predicted_links_world = self._candidate_tcp_world(candidate, observation)
        approach_axis = np.asarray(self.tro["approach_axis"], dtype=np.float64)
        approach_axis /= max(float(np.linalg.norm(approach_axis)), 1e-9)
        pre_tcp = final_tcp.copy()
        pre_tcp[:3, 3] += final_tcp[:3, :3] @ approach_axis * float(self.tro["pregrasp_offset_m"])
        lift_tcp = final_tcp.copy()
        lift_tcp[2, 3] += float(self.tro["lift_m"])

        start_arm = observation.arm_qpos[self.robot_index]
        open_q = observation.hand_qpos[self.robot_index, :12]
        pre = self.ik.solve_tcp(self.robot_index, start_arm, pre_tcp, hand_q=open_q)
        if not pre.converged:
            return None
        final = self.ik.solve_tcp(self.robot_index, pre.q, final_tcp, hand_q=open_q)
        if not final.converged:
            return None
        close_q = self._candidate_hand_q(candidate, final.q, open_q, predicted_links_world)
        lift = self.ik.solve_tcp(self.robot_index, final.q, lift_tcp, hand_q=close_q)
        if not lift.converged:
            return None

        durations = self.tro["stage_durations_s"]
        arm_segments = [
            _smooth_segment(start_arm, pre.q, round(durations["pregrasp"] * self.control_hz)),
            _smooth_segment(pre.q, final.q, round(durations["grasp"] * self.control_hz)),
            _smooth_segment(final.q, final.q, round(durations["close"] * self.control_hz)),
            _smooth_segment(final.q, lift.q, round(durations["lift"] * self.control_hz)),
            _smooth_segment(
                lift.q, self.robot["home_q"], round(durations["return"] * self.control_hz)
            ),
            _smooth_segment(
                self.robot["home_q"],
                self.robot["home_q"],
                round(durations["release"] * self.control_hz),
            ),
        ]
        hand_segments = [
            _smooth_segment(open_q, open_q, len(arm_segments[0])),
            _smooth_segment(open_q, open_q, len(arm_segments[1])),
            _smooth_segment(open_q, close_q, len(arm_segments[2])),
            _smooth_segment(close_q, close_q, len(arm_segments[3])),
            _smooth_segment(close_q, close_q, len(arm_segments[4])),
            _smooth_segment(close_q, open_q, len(arm_segments[5])),
        ]
        labels = ("pregrasp", "grasp", "close", "lift", "return", "release")
        stages = [
            label for label, segment in zip(labels, arm_segments) for _ in range(len(segment))
        ]
        info = {
            "candidate_rank": int(candidate.get("rank", 0)),
            "pregrasp_position_error_m": pre.position_error_m,
            "grasp_position_error_m": final.position_error_m,
            "lift_position_error_m": lift.position_error_m,
            "target_tcp_position": final_tcp[:3, 3].tolist(),
            "target_tcp_quaternion_wxyz": matrix_quaternion(final_tcp).tolist(),
        }
        return np.concatenate(arm_segments), np.concatenate(hand_segments), stages, info

    def reset(self, seed: int, initial_observation: PolicyObservation) -> None:
        del seed
        object_points, environment_points = self.pointclouds.build(
            initial_observation,
            target_object=self.target_object,
            robot=self.robot,
        )
        candidates = self.backend.infer(object_points, environment_points)
        if not candidates:
            raise RuntimeError("TRO returned no grasp candidates")
        for candidate in sorted(candidates, key=lambda item: int(item.get("rank", 0))):
            planned = self._plan_candidate(candidate, initial_observation)
            if planned is not None:
                selected_arm, selected_hand, self.stages, self.plan_info = planned
                robot_count = initial_observation.robot_count
                frame_count = len(selected_arm)
                self.arm_plan = np.repeat(initial_observation.arm_qpos[None], frame_count, axis=0)
                self.hand_plan = np.repeat(initial_observation.hand_qpos[None], frame_count, axis=0)
                self.arm_plan[:, self.robot_index] = selected_arm
                self.hand_plan[:, self.robot_index] = selected_hand
                assert self.arm_plan.shape == (frame_count, robot_count, 7)
                return
        raise RuntimeError(f"none of the {len(candidates)} TRO candidates was reachable")

    def act(self, observation: PolicyObservation) -> ActionChunk:
        if self.arm_plan is None or self.hand_plan is None:
            raise RuntimeError("TRO grasp policy was not reset")
        start = min(round(observation.time_s * self.control_hz), len(self.arm_plan) - 1)
        indices = np.clip(np.arange(start, start + self.action_horizon), 0, len(self.arm_plan) - 1)
        return ActionChunk(
            arm_q_target=self.arm_plan[indices],
            hand_q_target=self.hand_plan[indices],
            dt_s=self.dt_s,
            metadata={"stage": self.stages[start], **self.plan_info},
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "tro_grasp",
            "name": "tro_mujoco_ik_grasp_v1",
            "robot_id": self.robot["id"],
            "target_object": self.target_object,
            "control_hz": self.control_hz,
            "action_horizon": self.action_horizon,
            "grasp_constraint": dict(self.tro["grasp_constraint"]),
            "tro": self.backend.metadata(),
        }

    def close(self) -> None:
        self.backend.close()
