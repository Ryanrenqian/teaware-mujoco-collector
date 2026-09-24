from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from . import __version__
from .config import config_sha256, public_config
from .placement import PlacementError, TrayPlacement
from .policy import Policy, PolicyObservation, PolicyRunner, build_policy
from .robots import (
    ARM_ACTUATOR_SUFFIXES,
    ARM_JOINT_SUFFIXES,
    MAX_HAND_DOF,
    xhand_actuator_names,
    xhand_joint_names,
)
from .scene import write_scene_xml, yaw_quaternion
from .schema import SCHEMA_VERSION


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
        path, format="JPEG", quality=94, subsampling=0
    )


def _save_depth_preview(path: Path, depth: np.ndarray) -> None:
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size == 0:
        preview = np.zeros(depth.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [2.0, 98.0])
        if high <= low:
            high = low + 1e-6
        normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
        preview = np.asarray((1.0 - normalized) * 255.0, dtype=np.uint8)
    Image.fromarray(preview, mode="L").save(path)


def _segmentation_preview(labels: np.ndarray) -> np.ndarray:
    label_array = np.asarray(labels)
    object_ids = np.asarray(
        label_array[..., 0] if label_array.ndim == 3 else label_array,
        dtype=np.int64,
    )
    output = np.zeros((*object_ids.shape, 3), dtype=np.uint8)
    valid = object_ids >= 0
    ids = object_ids[valid].astype(np.uint64)
    output[..., 0][valid] = ((ids * 67 + 53) % 211 + 35).astype(np.uint8)
    output[..., 1][valid] = ((ids * 97 + 89) % 211 + 35).astype(np.uint8)
    output[..., 2][valid] = ((ids * 131 + 23) % 211 + 35).astype(np.uint8)
    return output


def _segmentation_from_idcolor(image: np.ndarray, scene: Any) -> np.ndarray:
    """Map MuJoCo ID-color pixels while treating invalid blended IDs as background."""
    image3 = np.asarray(image, dtype=np.uint32)
    segimage = image3[..., 0] + image3[..., 1] * (2**8) + image3[..., 2] * (2**16)
    output = np.full((*segimage.shape, 2), -1, dtype=np.int32)
    ngeoms = int(scene.ngeom)
    valid = segimage <= ngeoms
    if not np.any(valid):
        return output

    segid2output = np.full((ngeoms + 1, 2), -1, dtype=np.int32)
    visible_geoms = [geom for geom in scene.geoms[:ngeoms] if geom.segid != -1]
    for geom in visible_geoms:
        segid = int(geom.segid) + 1
        if 0 <= segid <= ngeoms:
            segid2output[segid] = (int(geom.objid), int(geom.objtype))
    output[valid] = segid2output[segimage[valid]]
    return output


class TeawareCollector:
    """Stateful MuJoCo scene and atomic episode writer."""

    def __init__(
        self,
        config: dict[str, Any],
        dataset_root: str | Path,
        policy: Policy | None = None,
    ):
        self.config = config
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.episodes_root = self.dataset_root / "episodes"
        self.runtime_root = self.dataset_root / ".runtime"
        self.episodes_root.mkdir(parents=True, exist_ok=True)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.scene_xml = write_scene_xml(config, self.runtime_root / "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)
        self.placement = TrayPlacement(self.model, config)
        self.width = int(config["renderer"]["width"])
        self.height = int(config["renderer"]["height"])
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.camera_names = [camera["name"] for camera in config["cameras"]]
        self.object_specs = list(config["objects"])
        self.object_names = [item["name"] for item in self.object_specs]
        self.object_body_ids = np.asarray(
            [self._name_id(mujoco.mjtObj.mjOBJ_BODY, name) for name in self.object_names],
            dtype=np.int32,
        )
        self.robots = [self._build_robot_runtime(spec) for spec in config["robots"]]
        self.policy = policy or build_policy(config, model=self.model, robots=self.robots)
        self._grasp_constraint = self._build_grasp_constraint()
        self._episode_max_finger_contacts = 0
        self._lock = threading.RLock()
        self._current_seed: int | None = None
        self._latest_frames: dict[str, dict[str, np.ndarray]] = {}
        self._write_dataset_metadata()

    def close(self) -> None:
        try:
            self.policy.close()
        finally:
            self.renderer.close()

    def status(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "cameras": list(self.camera_names),
            "objects": list(self.object_names),
            "scene_profile": self.config["scene_profile"],
            "robots": [
                {
                    "id": robot["id"],
                    "hand": robot["hand"],
                    "handedness": robot["handedness"],
                    "hand_dof": robot["hand_dof"],
                }
                for robot in self.robots
            ],
            "renderer": {"width": self.width, "height": self.height},
            "policy": self.policy.metadata(),
            "current_seed": self._current_seed,
        }

    def _name_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        value = mujoco.mj_name2id(self.model, object_type, name)
        if value < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return int(value)

    def _build_robot_runtime(self, spec: dict[str, Any]) -> dict[str, Any]:
        robot_id = spec["id"]
        arm_joint_names = [f"{robot_id}_{suffix}" for suffix in ARM_JOINT_SUFFIXES]
        arm_actuator_names = [f"{robot_id}_{suffix}" for suffix in ARM_ACTUATOR_SUFFIXES]
        arm_joint_ids = np.asarray(
            [self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in arm_joint_names],
            dtype=np.int32,
        )
        if spec["hand"] == "xhand":
            hand_joint_names = xhand_joint_names(robot_id, spec["handedness"])
            hand_actuator_names = xhand_actuator_names(robot_id)
            tcp_site_name = f"{robot_id}_hand_tcp"
        else:
            hand_joint_names = [f"{robot_id}_left_driver_joint"]
            hand_actuator_names = [f"{robot_id}_gripper"]
            tcp_site_name = f"{robot_id}_link_tcp"
        hand_joint_ids = np.asarray(
            [self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in hand_joint_names],
            dtype=np.int32,
        )
        arm_actuator_ids = np.asarray(
            [self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in arm_actuator_names],
            dtype=np.int32,
        )
        hand_actuator_ids = np.asarray(
            [self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in hand_actuator_names],
            dtype=np.int32,
        )
        return {
            "id": robot_id,
            "hand": spec["hand"],
            "handedness": spec["handedness"],
            "hand_dof": len(hand_joint_names),
            "arm_joint_names": arm_joint_names,
            "hand_joint_names": hand_joint_names,
            "arm_joint_ids": arm_joint_ids,
            "hand_joint_ids": hand_joint_ids,
            "arm_qpos_addresses": self.model.jnt_qposadr[arm_joint_ids].astype(np.int32),
            "arm_dof_addresses": self.model.jnt_dofadr[arm_joint_ids].astype(np.int32),
            "hand_qpos_addresses": self.model.jnt_qposadr[hand_joint_ids].astype(np.int32),
            "hand_dof_addresses": self.model.jnt_dofadr[hand_joint_ids].astype(np.int32),
            "arm_actuator_ids": arm_actuator_ids,
            "hand_actuator_ids": hand_actuator_ids,
            "tcp_site_id": self._name_id(mujoco.mjtObj.mjOBJ_SITE, tcp_site_name),
            "home_q": np.asarray(spec["home_q"], dtype=np.float64),
            "motion_amplitude": np.asarray(spec["motion_amplitude"], dtype=np.float64),
            "hand_home_q": np.asarray(spec["hand_home_q"], dtype=np.float64),
            "hand_motion_amplitude": np.asarray(
                spec["hand_motion_amplitude"], dtype=np.float64
            ),
            "base_position": list(spec["base_position"]),
            "base_yaw_deg": float(spec["base_yaw_deg"]),
        }

    def _build_grasp_constraint(self) -> dict[str, Any] | None:
        policy = self.config["policy"]
        if policy["type"] != "tro_grasp":
            return None
        constraint = policy["tro"]["grasp_constraint"]
        robot_index, robot = next(
            (index, runtime)
            for index, runtime in enumerate(self.robots)
            if runtime["id"] == policy["robot_id"]
        )
        hand_body_name = f"{robot['id']}_{robot['handedness']}_hand_link"
        hand_body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, hand_body_name)
        finger_prefix = f"{robot['id']}_{robot['handedness']}_hand_"
        finger_body_ids = {
            body_id
            for body_id in range(self.model.nbody)
            if (name := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id))
            and name.startswith(finger_prefix)
            and body_id != hand_body_id
        }
        object_body_id = self._name_id(
            mujoco.mjtObj.mjOBJ_BODY, policy["target_object"]
        )
        equality_id: int | None = None
        if constraint["enabled"]:
            equality_name = f"{robot['id']}_{policy['target_object']}_grasp_weld"
            equality_id = self._name_id(mujoco.mjtObj.mjOBJ_EQUALITY, equality_name)
        return {
            "enabled": bool(constraint["enabled"]),
            "equality_id": equality_id,
            "robot": robot,
            "robot_index": robot_index,
            "hand_body_id": hand_body_id,
            "finger_geom_ids": {
                geom_id
                for geom_id, body_id in enumerate(self.model.geom_bodyid)
                if int(body_id) in finger_body_ids
            },
            "object_geom_ids": {
                geom_id
                for geom_id, body_id in enumerate(self.model.geom_bodyid)
                if int(body_id) == object_body_id
            },
            "object_body_id": object_body_id,
            "max_distance_m": float(constraint["max_distance_m"]),
            "min_finger_contacts": int(constraint["min_finger_contacts"]),
            "min_closure_norm_rad": float(constraint["min_closure_norm_rad"]),
        }

    def _finger_contact_count(self, constraint: dict[str, Any]) -> int:
        finger_geom_ids = constraint["finger_geom_ids"]
        object_geom_ids = constraint["object_geom_ids"]
        contacting_bodies: set[int] = set()
        for contact in self.data.contact[: self.data.ncon]:
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in finger_geom_ids and geom2 in object_geom_ids:
                contacting_bodies.add(int(self.model.geom_bodyid[geom1]))
            elif geom2 in finger_geom_ids and geom1 in object_geom_ids:
                contacting_bodies.add(int(self.model.geom_bodyid[geom2]))
        return len(contacting_bodies)

    def _grasp_metrics(self, hand_targets: np.ndarray) -> dict[str, float | int | bool]:
        constraint = self._grasp_constraint
        if constraint is None:
            return {
                "finger_contacts": 0,
                "constraint_active": False,
                "closure_norm_rad": 0.0,
                "hand_tracking_rmse_rad": 0.0,
                "palm_object_distance_m": 0.0,
            }
        robot = constraint["robot"]
        actual = self.data.qpos[robot["hand_qpos_addresses"]]
        target = hand_targets[constraint["robot_index"], : robot["hand_dof"]]
        contacts = self._finger_contact_count(constraint)
        self._episode_max_finger_contacts = max(self._episode_max_finger_contacts, contacts)
        equality_id = constraint["equality_id"]
        return {
            "finger_contacts": contacts,
            "constraint_active": bool(
                equality_id is not None and self.data.eq_active[equality_id]
            ),
            "closure_norm_rad": float(np.linalg.norm(actual - robot["hand_home_q"])),
            "hand_tracking_rmse_rad": float(np.sqrt(np.mean(np.square(actual - target)))),
            "palm_object_distance_m": float(
                np.linalg.norm(
                    self.data.xpos[constraint["hand_body_id"]]
                    - self.data.xpos[constraint["object_body_id"]]
                )
            ),
        }

    def _update_grasp_constraint(self, stage: str, hand_targets: np.ndarray) -> None:
        constraint = self._grasp_constraint
        if constraint is None:
            return
        metrics = self._grasp_metrics(hand_targets)
        if not constraint["enabled"]:
            return
        equality_id = constraint["equality_id"]
        assert equality_id is not None
        if stage == "release":
            self.data.eq_active[equality_id] = 0
            return
        if stage != "close" or self.data.eq_active[equality_id]:
            return
        if metrics["palm_object_distance_m"] > constraint["max_distance_m"]:
            return
        if metrics["finger_contacts"] < constraint["min_finger_contacts"]:
            return
        if metrics["closure_norm_rad"] < constraint["min_closure_norm_rad"]:
            return
        hand_body_id = constraint["hand_body_id"]
        object_body_id = constraint["object_body_id"]
        inverse_position = np.empty(3, dtype=np.float64)
        inverse_quaternion = np.empty(4, dtype=np.float64)
        relative_position = np.empty(3, dtype=np.float64)
        relative_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_negPose(
            inverse_position,
            inverse_quaternion,
            self.data.xpos[hand_body_id],
            self.data.xquat[hand_body_id],
        )
        mujoco.mju_mulPose(
            relative_position,
            relative_quaternion,
            inverse_position,
            inverse_quaternion,
            self.data.xpos[object_body_id],
            self.data.xquat[object_body_id],
        )
        self.model.eq_data[equality_id, 3:6] = relative_position
        self.model.eq_data[equality_id, 6:10] = relative_quaternion
        self.data.eq_active[equality_id] = 1
        mujoco.mj_forward(self.model, self.data)

    def _write_dataset_metadata(self) -> None:
        path = self.dataset_root / "dataset.json"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "generator": {"name": "teaware-mujoco-collector", "version": __version__},
            "mujoco_version": mujoco.__version__,
            "config_sha256": config_sha256(self.config),
            "config": public_config(self.config),
            "policy": self.policy.metadata(),
        }
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("config_sha256") != payload["config_sha256"]:
                raise ValueError(
                    f"dataset {self.dataset_root} was created with a different configuration"
                )
            if existing.get("policy") != payload["policy"]:
                raise ValueError(
                    f"dataset {self.dataset_root} was created with a different policy"
                )
            return
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def _reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self._episode_max_finger_contacts = 0
        for robot in self.robots:
            self.data.qpos[robot["arm_qpos_addresses"]] = robot["home_q"]
            self.data.ctrl[robot["arm_actuator_ids"]] = robot["home_q"]
            self.data.qpos[robot["hand_qpos_addresses"]] = robot["hand_home_q"]
            self.data.ctrl[robot["hand_actuator_ids"]] = robot["hand_home_q"]
        mujoco.mj_forward(self.model, self.data)

    def _robot_state(self) -> tuple[np.ndarray, ...]:
        robot_count = len(self.robots)
        arm_qpos = np.zeros((robot_count, 7), dtype=np.float64)
        arm_qvel = np.zeros((robot_count, 7), dtype=np.float64)
        arm_ctrl = np.zeros((robot_count, 7), dtype=np.float64)
        hand_qpos = np.zeros((robot_count, MAX_HAND_DOF), dtype=np.float64)
        hand_qvel = np.zeros((robot_count, MAX_HAND_DOF), dtype=np.float64)
        hand_ctrl = np.zeros((robot_count, MAX_HAND_DOF), dtype=np.float64)
        tcp_position = np.zeros((robot_count, 3), dtype=np.float64)
        tcp_quaternion = np.zeros((robot_count, 4), dtype=np.float64)
        for index, robot in enumerate(self.robots):
            hand_dof = robot["hand_dof"]
            arm_qpos[index] = self.data.qpos[robot["arm_qpos_addresses"]]
            arm_qvel[index] = self.data.qvel[robot["arm_dof_addresses"]]
            arm_ctrl[index] = self.data.ctrl[robot["arm_actuator_ids"]]
            hand_qpos[index, :hand_dof] = self.data.qpos[robot["hand_qpos_addresses"]]
            hand_qvel[index, :hand_dof] = self.data.qvel[robot["hand_dof_addresses"]]
            hand_ctrl[index, :hand_dof] = self.data.ctrl[robot["hand_actuator_ids"]]
            site_id = robot["tcp_site_id"]
            tcp_position[index] = self.data.site_xpos[site_id]
            mujoco.mju_mat2Quat(tcp_quaternion[index], self.data.site_xmat[site_id])
        return (
            arm_qpos,
            arm_qvel,
            arm_ctrl,
            hand_qpos,
            hand_qvel,
            hand_ctrl,
            tcp_position,
            tcp_quaternion,
        )

    def _policy_observation(
        self,
        *,
        episode_id: str,
        step_index: int,
        time_s: float,
        modalities: dict[str, dict[str, np.ndarray]],
        robot_state: tuple[np.ndarray, ...],
    ) -> PolicyObservation:
        return PolicyObservation(
            episode_id=episode_id,
            step_index=int(step_index),
            time_s=float(time_s),
            task=self.config["policy"]["task"],
            robot_ids=tuple(robot["id"] for robot in self.robots),
            hand_dof=np.asarray([robot["hand_dof"] for robot in self.robots], dtype=np.int32),
            arm_qpos=robot_state[0].copy(),
            arm_qvel=robot_state[1].copy(),
            hand_qpos=robot_state[3].copy(),
            hand_qvel=robot_state[4].copy(),
            tcp_position=robot_state[6].copy(),
            tcp_quaternion=robot_state[7].copy(),
            object_names=tuple(self.object_names),
            object_position=self.data.xpos[self.object_body_ids].copy(),
            object_quaternion=self.data.xquat[self.object_body_ids].copy(),
            images_rgb={name: images["rgb"].copy() for name, images in modalities.items()},
            images_depth={name: images["depth"].copy() for name, images in modalities.items()},
        )

    def _apply_policy_targets(
        self,
        arm_targets: np.ndarray,
        hand_targets: np.ndarray,
    ) -> None:
        for index, robot in enumerate(self.robots):
            arm_ids = robot["arm_actuator_ids"]
            arm_limits = self.model.actuator_ctrlrange[arm_ids]
            self.data.ctrl[arm_ids] = np.clip(
                arm_targets[index], arm_limits[:, 0], arm_limits[:, 1]
            )
            hand_ids = robot["hand_actuator_ids"]
            hand_limits = self.model.actuator_ctrlrange[hand_ids]
            hand_dof = robot["hand_dof"]
            self.data.ctrl[hand_ids] = np.clip(
                hand_targets[index, :hand_dof],
                hand_limits[:, 0],
                hand_limits[:, 1],
            )

    def _sample_placements(self, rng: np.random.Generator) -> list[tuple[float, float, float]]:
        return self.placement.sample(rng)

    def randomize(self, seed: int) -> dict[str, Any]:
        with self._lock:
            rng = np.random.default_rng(int(seed))
            self._current_seed = None
            self._latest_frames = {}
            attempts = self.config["randomization"]["max_scene_attempts"]
            last_issues = []
            for attempt in range(1, attempts + 1):
                self._reset()
                try:
                    placements = self._sample_placements(rng)
                except PlacementError as exc:
                    last_issues = [str(exc)]
                    continue
                sampled: dict[str, Any] = {}
                for index, (spec, (x, y, yaw)) in enumerate(zip(self.object_specs, placements)):
                    joint_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, f"{spec['name']}_free")
                    qpos_address = int(self.model.jnt_qposadr[joint_id])
                    position = [x, y, self.placement.spawn_height(index)]
                    self.data.qpos[qpos_address : qpos_address + 3] = position
                    self.data.qpos[qpos_address + 3 : qpos_address + 7] = yaw_quaternion(yaw)
                    sampled[spec["name"]] = {"position": position, "yaw_deg": yaw}
                mujoco.mj_forward(self.model, self.data)
                report = self.placement.settle(
                    self.data,
                    self.config["simulation"]["settle_s"],
                    self.config["simulation"]["settle_timeout_s"],
                )
                if not report["valid"]:
                    last_issues = report["issues"]
                    continue
                self._latest_frames = self._render_all()
                self._current_seed = int(seed)
                return {
                    "seed": int(seed), "placements": sampled, "scene_attempts": attempt,
                    "placement_surface": "tea_tray", "bounds_method": "rotated_body_aabb",
                    "edge_margin_m": self.config["randomization"]["edge_margin_m"],
                    "settled": report,
                }
            raise PlacementError(
                f"no valid tea_tray layout for seed {seed} after {attempts} scene attempts: "
                + "; ".join(last_issues)
            )

    def _render_camera(self, camera_name: str) -> dict[str, np.ndarray]:
        self.renderer.disable_depth_rendering()
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=camera_name)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera_name)
        depth = self.renderer.render().astype(np.float32, copy=True)
        self.renderer.disable_depth_rendering()
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=camera_name)
        try:
            segmentation = self.renderer.render().astype(np.int32, copy=True)
        except IndexError:
            # MuJoCo <= 3.8 can emit blended ID colors outside the segid table.
            # The framebuffer is still valid, so remap it with explicit bounds checks.
            renderer = self.renderer
            scene = renderer._scene
            pixels = np.empty((self.height, self.width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(pixels, None, renderer._rect, renderer._mjr_context)
            segmentation = _segmentation_from_idcolor(pixels, scene)
            if renderer._gl_context:
                segmentation = np.flipud(segmentation)
            scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = False
            scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = False
        self.renderer.disable_segmentation_rendering()
        instance = np.full(segmentation.shape[:2], -1, dtype=np.int32)
        geom_pixels = (
            (segmentation[..., 0] >= 0)
            & (segmentation[..., 0] < self.model.ngeom)
            & (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        )
        instance[geom_pixels] = self.model.geom_bodyid[segmentation[..., 0][geom_pixels]]
        return {
            "rgb": rgb,
            "depth": depth,
            "segmentation": segmentation,
            "instance": instance,
        }

    def _render_all(self) -> dict[str, dict[str, np.ndarray]]:
        return {name: self._render_camera(name) for name in self.camera_names}

    def latest_frames(self) -> dict[str, dict[str, np.ndarray]]:
        with self._lock:
            if not self._latest_frames:
                self.randomize(0)
            return {
                camera: {modality: array.copy() for modality, array in modalities.items()}
                for camera, modalities in self._latest_frames.items()
            }

    def _camera_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for name in self.camera_names:
            camera_id = self._name_id(mujoco.mjtObj.mjOBJ_CAMERA, name)
            fovy = float(self.model.cam_fovy[camera_id])
            fy = 0.5 * self.height / np.tan(np.deg2rad(fovy) / 2.0)
            fx = fy
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = self.data.cam_xmat[camera_id].reshape(3, 3)
            transform[:3, 3] = self.data.cam_xpos[camera_id]
            metadata[name] = {
                "width": self.width,
                "height": self.height,
                "fovy_deg": fovy,
                "intrinsics": [
                    [float(fx), 0.0, (self.width - 1) / 2.0],
                    [0.0, float(fy), (self.height - 1) / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                "T_world_camera": transform.tolist(),
                "T_camera_world": np.linalg.inv(transform).tolist(),
                "convention": "MuJoCo camera: +X right, +Y up, optical axis -Z",
            }
        return metadata

    def _label_map(self) -> dict[str, Any]:
        labels: dict[str, Any] = {}
        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            labels[str(geom_id)] = {"geom": geom_name, "body": body_name}
        return labels

    def _instance_label_map(self) -> dict[str, str]:
        return {
            str(body_id): mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or f"body_{body_id}"
            for body_id in range(self.model.nbody)
        }

    def _next_episode_id(self) -> str:
        indexes = []
        for path in self.episodes_root.glob("episode_[0-9]*"):
            suffix = path.name.removeprefix("episode_")
            if suffix.isdigit():
                indexes.append(int(suffix))
        return f"episode_{(max(indexes) + 1) if indexes else 0:06d}"

    def collect_episode(self, seed: int) -> Path:
        with self._lock:
            randomization = self.randomize(seed)
            episode_id = self._next_episode_id()
            final_dir = self.episodes_root / episode_id
            tmp_dir = self.episodes_root / f".{episode_id}.tmp-{uuid.uuid4().hex}"
            frame_root = tmp_dir / "frames"
            frame_root.mkdir(parents=True)
            try:
                result = self._capture(tmp_dir, frame_root, episode_id, int(seed), randomization)
                os.replace(tmp_dir, final_dir)
                self._append_index(result)
                return final_dir
            except Exception:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                raise

    def _capture(
        self,
        tmp_dir: Path,
        frame_root: Path,
        episode_id: str,
        seed: int,
        randomization: dict[str, Any],
    ) -> dict[str, Any]:
        simulation = self.config["simulation"]
        duration = float(simulation["duration_s"])
        capture_fps = float(simulation["capture_fps"])
        frame_times = np.arange(0.0, duration + 0.5 / capture_fps, 1.0 / capture_fps)
        start_time = float(self.data.time)
        policy_runner = PolicyRunner(self.policy, len(self.robots))

        time_rows: list[float] = []
        qpos_rows: list[np.ndarray] = []
        qvel_rows: list[np.ndarray] = []
        ctrl_rows: list[np.ndarray] = []
        body_pos_rows: list[np.ndarray] = []
        body_quat_rows: list[np.ndarray] = []
        arm_qpos_rows: list[np.ndarray] = []
        arm_qvel_rows: list[np.ndarray] = []
        arm_ctrl_rows: list[np.ndarray] = []
        hand_qpos_rows: list[np.ndarray] = []
        hand_qvel_rows: list[np.ndarray] = []
        hand_ctrl_rows: list[np.ndarray] = []
        tcp_position_rows: list[np.ndarray] = []
        tcp_quaternion_rows: list[np.ndarray] = []
        policy_arm_target_rows: list[np.ndarray] = []
        policy_hand_target_rows: list[np.ndarray] = []
        policy_latency_rows: list[float] = []
        policy_horizon_rows: list[int] = []
        policy_mode_rows: list[str] = []
        policy_stage_rows: list[str] = []
        finger_contact_rows: list[int] = []
        grasp_constraint_active_rows: list[bool] = []
        hand_closure_norm_rows: list[float] = []
        hand_tracking_rmse_rows: list[float] = []
        palm_object_distance_rows: list[float] = []
        frame_records: list[dict[str, Any]] = []

        for frame_index, relative_time in enumerate(frame_times):
            target_time = start_time + float(relative_time)
            while self.data.time + self.model.opt.timestep * 0.5 < target_time:
                elapsed = float(self.data.time - start_time)
                arm_targets, hand_targets = policy_runner.targets_at(elapsed)
                self._apply_policy_targets(arm_targets, hand_targets)
                stage = (
                    str(policy_runner.chunk.metadata.get("stage", ""))
                    if policy_runner.chunk is not None
                    else ""
                )
                self._update_grasp_constraint(stage, hand_targets)
                mujoco.mj_step(self.model, self.data)

            modalities = self._render_all()
            self._latest_frames = modalities
            observation_state = self._robot_state()
            observation = self._policy_observation(
                episode_id=episode_id,
                step_index=frame_index,
                time_s=float(relative_time),
                modalities=modalities,
                robot_state=observation_state,
            )
            if frame_index == 0:
                policy_runner.reset(seed, observation)
            chunk = policy_runner.submit(observation)
            arm_targets, hand_targets = policy_runner.targets_at(float(relative_time))
            self._apply_policy_targets(arm_targets, hand_targets)
            self._update_grasp_constraint(str(chunk.metadata.get("stage", "")), hand_targets)
            grasp_metrics = self._grasp_metrics(hand_targets)
            frame_id = f"{frame_index:06d}"
            frame_dir = frame_root / frame_id
            frame_dir.mkdir()
            for camera_name, images in modalities.items():
                prefix = frame_dir / camera_name
                _save_rgb(prefix.with_name(prefix.name + "_rgb.jpg"), images["rgb"])
                np.save(
                    prefix.with_name(prefix.name + "_depth.npy"),
                    images["depth"],
                    allow_pickle=False,
                )
                np.save(
                    prefix.with_name(prefix.name + "_segmentation.npy"),
                    images["segmentation"],
                    allow_pickle=False,
                )
                np.save(
                    prefix.with_name(prefix.name + "_instance.npy"),
                    images["instance"],
                    allow_pickle=False,
                )
                _save_depth_preview(
                    prefix.with_name(prefix.name + "_depth_preview.png"), images["depth"]
                )
                Image.fromarray(_segmentation_preview(images["instance"]), mode="RGB").save(
                    prefix.with_name(prefix.name + "_segmentation_preview.png")
                )
            time_rows.append(float(self.data.time - start_time))
            qpos_rows.append(self.data.qpos.copy())
            qvel_rows.append(self.data.qvel.copy())
            ctrl_rows.append(self.data.ctrl.copy())
            body_pos_rows.append(self.data.xpos[self.object_body_ids].copy())
            body_quat_rows.append(self.data.xquat[self.object_body_ids].copy())
            robot_state = self._robot_state()
            arm_qpos_rows.append(robot_state[0])
            arm_qvel_rows.append(robot_state[1])
            arm_ctrl_rows.append(robot_state[2])
            hand_qpos_rows.append(robot_state[3])
            hand_qvel_rows.append(robot_state[4])
            hand_ctrl_rows.append(robot_state[5])
            tcp_position_rows.append(robot_state[6])
            tcp_quaternion_rows.append(robot_state[7])
            policy_arm_target_rows.append(arm_targets.copy())
            policy_hand_target_rows.append(hand_targets.copy())
            policy_latency_rows.append(policy_runner.last_latency_ms)
            policy_horizon_rows.append(chunk.horizon)
            policy_mode_rows.append(chunk.action_mode)
            policy_stage_rows.append(str(chunk.metadata.get("stage", "")))
            finger_contact_rows.append(int(grasp_metrics["finger_contacts"]))
            grasp_constraint_active_rows.append(bool(grasp_metrics["constraint_active"]))
            hand_closure_norm_rows.append(float(grasp_metrics["closure_norm_rad"]))
            hand_tracking_rmse_rows.append(float(grasp_metrics["hand_tracking_rmse_rad"]))
            palm_object_distance_rows.append(float(grasp_metrics["palm_object_distance_m"]))
            frame_records.append(
                {
                    "frame_id": frame_id,
                    "time_s": time_rows[-1],
                    "policy_latency_ms": policy_latency_rows[-1],
                    "policy_action_horizon": chunk.horizon,
                }
            )

        trajectory_path = tmp_dir / "trajectory.npz"
        np.savez_compressed(
            trajectory_path,
            time=np.asarray(time_rows, dtype=np.float64),
            qpos=np.asarray(qpos_rows, dtype=np.float64),
            qvel=np.asarray(qvel_rows, dtype=np.float64),
            ctrl=np.asarray(ctrl_rows, dtype=np.float64),
            body_position=np.asarray(body_pos_rows, dtype=np.float64),
            body_quaternion=np.asarray(body_quat_rows, dtype=np.float64),
            body_names=np.asarray(self.object_names),
            arm_qpos=np.asarray(arm_qpos_rows, dtype=np.float64),
            arm_qvel=np.asarray(arm_qvel_rows, dtype=np.float64),
            arm_ctrl=np.asarray(arm_ctrl_rows, dtype=np.float64),
            hand_qpos=np.asarray(hand_qpos_rows, dtype=np.float64),
            hand_qvel=np.asarray(hand_qvel_rows, dtype=np.float64),
            hand_ctrl=np.asarray(hand_ctrl_rows, dtype=np.float64),
            tcp_position=np.asarray(tcp_position_rows, dtype=np.float64),
            tcp_quaternion=np.asarray(tcp_quaternion_rows, dtype=np.float64),
            policy_arm_q_target=np.asarray(policy_arm_target_rows, dtype=np.float64),
            policy_hand_q_target=np.asarray(policy_hand_target_rows, dtype=np.float64),
            policy_latency_ms=np.asarray(policy_latency_rows, dtype=np.float64),
            policy_action_horizon=np.asarray(policy_horizon_rows, dtype=np.int32),
            policy_action_mode=np.asarray(policy_mode_rows),
            policy_stage=np.asarray(policy_stage_rows),
            grasp_finger_contacts=np.asarray(finger_contact_rows, dtype=np.int32),
            grasp_constraint_active=np.asarray(grasp_constraint_active_rows, dtype=np.bool_),
            hand_closure_norm_rad=np.asarray(hand_closure_norm_rows, dtype=np.float64),
            hand_tracking_rmse_rad=np.asarray(hand_tracking_rmse_rows, dtype=np.float64),
            palm_object_distance_m=np.asarray(palm_object_distance_rows, dtype=np.float64),
            robot_ids=np.asarray([robot["id"] for robot in self.robots]),
            hand_types=np.asarray([robot["hand"] for robot in self.robots]),
            hand_dof=np.asarray([robot["hand_dof"] for robot in self.robots], dtype=np.int32),
        )
        grasp_outcome: dict[str, Any] | None = None
        if self._grasp_constraint is not None:
            target_index = self.object_names.index(self.config["policy"]["target_object"])
            target_z = np.asarray(body_pos_rows, dtype=np.float64)[:, target_index, 2]
            lift_delta_m = float(np.max(target_z) - target_z[0])
            assisted = bool(np.any(grasp_constraint_active_rows))
            min_contacts = 1 if assisted else 2
            grasp_outcome = {
                "success": bool(
                    lift_delta_m >= 0.05
                    and self._episode_max_finger_contacts >= min_contacts
                ),
                "assisted": assisted,
                "lift_delta_m": lift_delta_m,
                "max_finger_contacts": int(self._episode_max_finger_contacts),
                "max_hand_closure_norm_rad": float(max(hand_closure_norm_rows, default=0.0)),
                "max_hand_tracking_rmse_rad": float(max(hand_tracking_rmse_rows, default=0.0)),
            }

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "created_at": _utc_now(),
            "seed": seed,
            "generator": {"name": "teaware-mujoco-collector", "version": __version__},
            "mujoco_version": mujoco.__version__,
            "config_sha256": config_sha256(self.config),
            "scene_xml_sha256": _sha256_file(self.scene_xml),
            "coordinate_frame": "MuJoCo world, metres, quaternion wxyz",
            "renderer": {"width": self.width, "height": self.height},
            "cameras": self._camera_metadata(),
            "objects": self.object_names,
            "scene_profile": self.config["scene_profile"],
            "policy": {
                **self.policy.metadata(),
                "task": self.config["policy"]["task"],
            },
            "robots": [
                {
                    "id": robot["id"],
                    "hand_type": robot["hand"],
                    "handedness": robot["handedness"],
                    "base_position": robot["base_position"],
                    "base_yaw_deg": robot["base_yaw_deg"],
                    "arm_joint_names": robot["arm_joint_names"],
                    "hand_joint_names": robot["hand_joint_names"],
                    "hand_dof": robot["hand_dof"],
                }
                for robot in self.robots
            ],
            "segmentation": {
                "raw_array": "*_segmentation.npy",
                "raw_array_channels": ["object_id", "object_type"],
                "geom_label_map": self._label_map(),
                "instance_array": "*_instance.npy",
                "instance_definition": "MuJoCo body id; -1 is background",
                "body_label_map": self._instance_label_map(),
            },
            "randomization": randomization,
            **({"grasp_outcome": grasp_outcome} if grasp_outcome is not None else {}),
            "frames": frame_records,
            "trajectory": {
                "path": "trajectory.npz",
                "arrays": {
                    "time": "(T,) seconds from episode start",
                    "qpos": "(T, nq)",
                    "qvel": "(T, nv)",
                    "ctrl": "(T, nu)",
                    "body_position": "(T, object, 3) world metres",
                    "body_quaternion": "(T, object, 4) world wxyz",
                    "body_names": "(object,)",
                    "arm_qpos": "(T, robot, 7)",
                    "arm_qvel": "(T, robot, 7)",
                    "arm_ctrl": "(T, robot, 7)",
                    "hand_qpos": "(T, robot, 12), zero-padded after hand_dof",
                    "hand_qvel": "(T, robot, 12), zero-padded after hand_dof",
                    "hand_ctrl": "(T, robot, 12), zero-padded after hand_dof",
                    "tcp_position": "(T, robot, 3) world metres",
                    "tcp_quaternion": "(T, robot, 4) world wxyz",
                    "policy_arm_q_target": "(T, robot, 7)",
                    "policy_hand_q_target": "(T, robot, 12)",
                    "policy_latency_ms": "(T,)",
                    "policy_action_horizon": "(T,)",
                    "policy_action_mode": "(T,)",
                    "policy_stage": "(T,)",
                    "grasp_finger_contacts": "(T,) distinct contacting finger links",
                    "grasp_constraint_active": "(T,) boolean",
                    "hand_closure_norm_rad": "(T,) L2 distance from open hand pose",
                    "hand_tracking_rmse_rad": "(T,) joint target tracking RMSE",
                    "palm_object_distance_m": "(T,) metres",
                    "robot_ids": "(robot,)",
                    "hand_types": "(robot,)",
                    "hand_dof": "(robot,)",
                },
            },
        }
        manifest_path = tmp_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return {
            "episode_id": episode_id,
            "created_at": manifest["created_at"],
            "seed": seed,
            "frames": len(frame_records),
            "path": f"episodes/{episode_id}",
        }

    def _append_index(self, record: dict[str, Any]) -> None:
        index_path = self.dataset_root / "dataset.jsonl"
        with index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def collect(self, episodes: int, seed: int) -> list[Path]:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        return [self.collect_episode(seed + index) for index in range(episodes)]
