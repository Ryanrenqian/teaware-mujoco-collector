from __future__ import annotations

from itertools import product
from typing import Any

import mujoco
import numpy as np


class PlacementError(RuntimeError):
    """No valid initial layout was found within the configured attempt budget."""


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def _corners(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.array(list(product(*zip(low, high))))


def _body_bounds(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    """Conservative body-frame corners including visual AND collision geometry."""
    parts = []
    for geom_id in np.flatnonzero(model.geom_bodyid == body_id):
        kind = model.geom_type[geom_id]
        size = model.geom_size[geom_id]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom_id]
            start = model.mesh_vertadr[mesh_id]
            points = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id]]
        else:
            if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                half = np.repeat(size[0], 3)
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                half = np.array([size[0], size[0], size[0] + size[1]])
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                half = np.array([size[0], size[0], size[1]])
            elif kind in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                half = size
            else:
                raise PlacementError(f"unsupported teaware geom type: {kind}")
            points = _corners(-half, half)
        # MuJoCo recentres/reorients compiled meshes; include the compiled geom pose.
        parts.append(points @ _rotation(model.geom_quat[geom_id]).T + model.geom_pos[geom_id])
    if not parts:
        raise PlacementError(f"teaware body {body_id} has no geometry")
    vertices = np.concatenate(parts)
    return _corners(vertices.min(axis=0), vertices.max(axis=0))


def _xy_bounds(points: np.ndarray) -> np.ndarray:
    return np.array([points[:, :2].min(axis=0), points[:, :2].max(axis=0)])


def _box_gap(first: np.ndarray, second: np.ndarray) -> float:
    separation = np.maximum(0.0, np.maximum(first[0] - second[1], second[0] - first[1]))
    return float(np.linalg.norm(separation))


class TrayPlacement:
    """Place free-joint teaware inside the actual compiled green tray."""

    def __init__(self, model: mujoco.MjModel, config: dict[str, Any]):
        self.model = model
        self.specs = config["objects"]
        self.settings = config["randomization"]
        self.tray_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tea_tray")
        if self.tray_id < 0:
            raise PlacementError("tea_tray geom is required for teaware placement")
        if (
            model.geom_type[self.tray_id] != mujoco.mjtGeom.mjGEOM_BOX
            or model.geom_bodyid[self.tray_id] != 0
            or not np.allclose(_rotation(model.geom_quat[self.tray_id]), np.eye(3))
        ):
            raise PlacementError("tea_tray must be a fixed, axis-aligned box")
        center = model.geom_pos[self.tray_id]
        half = model.geom_size[self.tray_id]
        self.tray_bounds = np.array([center[:2] - half[:2], center[:2] + half[:2]])
        self.safe_bounds = (
            self.tray_bounds + np.array([[1.0], [-1.0]]) * self.settings["edge_margin_m"]
        )
        if np.any(self.safe_bounds[0] >= self.safe_bounds[1]):
            raise PlacementError("edge_margin_m leaves no usable area on tea_tray")
        self.support_z = float(center[2] + half[2])
        self.body_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec["name"])
                for spec in self.specs
            ]
        )
        if np.any(self.body_ids < 0):
            raise PlacementError("a configured teaware body is missing")
        self.corners = [_body_bounds(model, int(body_id)) for body_id in self.body_ids]
        # Place bulky vessels first, but return poses in the public YAML object order.
        areas = [float(np.prod(np.ptp(points[:, :2], axis=0))) for points in self.corners]
        self.order = sorted(range(len(self.specs)), key=lambda i: -areas[i])

    def spawn_height(self, index: int) -> float:
        return self.support_z - float(self.corners[index][:, 2].min()) + 0.025

    def sample(self, rng: np.random.Generator) -> list[tuple[float, float, float]]:
        poses: dict[int, tuple[float, float, float]] = {}
        bounds: dict[int, np.ndarray] = {}
        for index in self.order:
            spec = self.specs[index]
            ranges = spec["randomization"]
            for _ in range(self.settings["max_placement_attempts"]):
                yaw = float(rng.uniform(*ranges["yaw_deg"]))
                angle = np.deg2rad(yaw) / 2.0
                rotation = _rotation(np.array([np.cos(angle), 0.0, 0.0, np.sin(angle)]))
                offsets = _xy_bounds(self.corners[index] @ rotation.T)
                # Intersect the YAML centre range with the range that fits this orientation.
                low = np.maximum([ranges["x"][0], ranges["y"][0]], self.safe_bounds[0] - offsets[0])
                high = np.minimum(
                    [ranges["x"][1], ranges["y"][1]], self.safe_bounds[1] - offsets[1]
                )
                if np.any(low > high):
                    continue
                xy = rng.uniform(low, high)
                candidate = offsets + xy
                if any(
                    np.linalg.norm(xy - poses[other][:2]) < self.settings["minimum_object_distance"]
                    or _box_gap(candidate, bounds[other]) < self.settings["minimum_object_gap_m"]
                    for other in poses
                ):
                    continue
                poses[index] = (float(xy[0]), float(xy[1]), yaw)
                bounds[index] = candidate
                break
            else:
                raise PlacementError(
                    f"cannot place {spec['name']!r} inside tea_tray with "
                    f"edge_margin_m={self.settings['edge_margin_m']}; "
                    "check its size, yaw/XY ranges and object spacing"
                )
        return [poses[index] for index in range(len(self.specs))]

    def inspect(self, data: mujoco.MjData) -> dict[str, Any]:
        """Check actual poses, support contacts and rest velocities after stepping."""
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            return {"valid": False, "issues": ["non-finite simulation state"], "objects": {}}
        supported = set()
        forbidden = set()
        objects = set(self.body_ids)
        for contact in data.contact[: data.ncon]:
            if contact.efc_address < 0:
                continue
            for own, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                body_id = int(self.model.geom_bodyid[own])
                if body_id not in objects:
                    continue
                if other == self.tray_id:
                    supported.add(body_id)
                elif self.model.geom_bodyid[other] != body_id:
                    forbidden.add(body_id)
        issues = []
        records = {}
        bounds = []
        for index, (spec, body_id) in enumerate(zip(self.specs, self.body_ids)):
            rotation = data.xmat[body_id].reshape(3, 3)
            box = _xy_bounds(self.corners[index] @ rotation.T + data.xpos[body_id])
            bounds.append(box)
            clearance = float(
                min(np.min(box[0] - self.tray_bounds[0]), np.min(self.tray_bounds[1] - box[1]))
            )
            tilt = float(np.rad2deg(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(
                self.model, data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), velocity, 0
            )
            linear_speed = float(np.linalg.norm(velocity[3:]))
            angular_speed = float(np.linalg.norm(velocity[:3]))
            name = spec["name"]
            records[name] = {
                "position": data.xpos[body_id].tolist(),
                "quaternion_wxyz": data.xquat[body_id].tolist(),
                "edge_clearance_m": clearance,
                "tilt_deg": tilt,
                "linear_speed_m_s": linear_speed,
                "angular_speed_rad_s": angular_speed,
                "supported_by_tray": body_id in supported,
            }
            if clearance + 1e-9 < self.settings["edge_margin_m"]:
                issues.append(f"{name}: tray edge clearance {clearance:.4f} m")
            if body_id not in supported:
                issues.append(f"{name}: no tray support contact")
            if body_id in forbidden:
                issues.append(f"{name}: contact outside tray (robot, floor or another object)")
            if tilt > 10.0 or linear_speed > 0.02 or angular_speed > 0.2:
                issues.append(f"{name}: not upright and at rest")
            for other in range(index):
                center_distance = np.linalg.norm(
                    data.xpos[body_id, :2] - data.xpos[self.body_ids[other], :2]
                )
                if (
                    _box_gap(box, bounds[other]) + 1e-9 < self.settings["minimum_object_gap_m"]
                    or center_distance + 1e-9 < self.settings["minimum_object_distance"]
                ):
                    issues.append(f"{name}/{self.specs[other]['name']}: insufficient spacing")
        return {"valid": not issues, "issues": issues, "objects": records}

    def settle(self, data: mujoco.MjData, minimum_s: float, timeout_s: float) -> dict[str, Any]:
        dt = self.model.opt.timestep
        minimum_steps = int(np.ceil(minimum_s / dt))
        timeout_steps = int(np.ceil(timeout_s / dt))
        interval = max(1, round(0.02 / dt))
        stable_since: int | None = None
        steps = 0
        report: dict[str, Any] = {"valid": False, "issues": ["settling timeout"]}
        while steps < timeout_steps:
            batch = min(interval, timeout_steps - steps)
            mujoco.mj_step(self.model, data, nstep=batch)
            steps += batch
            mujoco.mj_forward(self.model, data)
            report = self.inspect(data)
            if not report["valid"]:
                stable_since = None
            elif stable_since is None:
                stable_since = steps
            if (
                steps >= minimum_steps
                and stable_since is not None
                and (steps - stable_since) * dt >= 0.1
            ):
                return {**report, "settle_time_s": steps * dt}
        return {
            **report,
            "valid": False,
            "settle_time_s": steps * dt,
            "issues": report["issues"] or ["not stable for 0.1 s before settling timeout"],
        }
