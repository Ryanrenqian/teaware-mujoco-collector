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
from .scene import object_spawn_height, table_top_z, write_scene_xml, yaw_quaternion
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


class TeawareCollector:
    """Stateful MuJoCo scene and atomic episode writer."""

    def __init__(self, config: dict[str, Any], dataset_root: str | Path):
        self.config = config
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.episodes_root = self.dataset_root / "episodes"
        self.runtime_root = self.dataset_root / ".runtime"
        self.episodes_root.mkdir(parents=True, exist_ok=True)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.scene_xml = write_scene_xml(config, self.runtime_root / "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)
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
        self.arm_actuator_ids = np.asarray(
            [self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"act{index}") for index in range(1, 8)],
            dtype=np.int32,
        )
        self._lock = threading.RLock()
        self._current_seed: int | None = None
        self._latest_frames: dict[str, dict[str, np.ndarray]] = {}
        self._write_dataset_metadata()

    def close(self) -> None:
        self.renderer.close()

    def status(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "cameras": list(self.camera_names),
            "objects": list(self.object_names),
            "renderer": {"width": self.width, "height": self.height},
            "current_seed": self._current_seed,
        }

    def _name_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        value = mujoco.mj_name2id(self.model, object_type, name)
        if value < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return int(value)

    def _write_dataset_metadata(self) -> None:
        path = self.dataset_root / "dataset.json"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "generator": {"name": "teaware-mujoco-collector", "version": __version__},
            "mujoco_version": mujoco.__version__,
            "config_sha256": config_sha256(self.config),
            "config": public_config(self.config),
        }
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("config_sha256") != payload["config_sha256"]:
                raise ValueError(
                    f"dataset {self.dataset_root} was created with a different configuration"
                )
            return
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def _reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        home = np.asarray(self.config["robot"]["home_q"], dtype=np.float64)
        self.data.ctrl[self.arm_actuator_ids] = home

    def _sample_placements(self, rng: np.random.Generator) -> list[tuple[float, float, float]]:
        minimum_distance = float(self.config["randomization"]["minimum_object_distance"])
        max_attempts = int(self.config["randomization"]["max_placement_attempts"])
        placements: list[tuple[float, float, float]] = []
        for spec in self.object_specs:
            ranges = spec["randomization"]
            for _ in range(max_attempts):
                x = float(rng.uniform(*ranges["x"]))
                y = float(rng.uniform(*ranges["y"]))
                if all(
                    np.hypot(x - old_x, y - old_y) >= minimum_distance
                    for old_x, old_y, _ in placements
                ):
                    yaw = float(rng.uniform(*ranges["yaw_deg"]))
                    placements.append((x, y, yaw))
                    break
            else:
                raise RuntimeError(
                    f"could not place {spec['name']!r}; relax minimum_object_distance or ranges"
                )
        return placements

    def randomize(self, seed: int) -> dict[str, Any]:
        with self._lock:
            rng = np.random.default_rng(int(seed))
            self._reset()
            placements = self._sample_placements(rng)
            support_z = table_top_z(self.config) + 0.016
            sampled: dict[str, Any] = {}
            for spec, (x, y, yaw) in zip(self.object_specs, placements):
                joint_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, f"{spec['name']}_free")
                qpos_address = int(self.model.jnt_qposadr[joint_id])
                position = [x, y, object_spawn_height(spec["preset"], support_z)]
                self.data.qpos[qpos_address : qpos_address + 3] = position
                self.data.qpos[qpos_address + 3 : qpos_address + 7] = yaw_quaternion(yaw)
                sampled[spec["name"]] = {"position": position, "yaw_deg": yaw}
            mujoco.mj_forward(self.model, self.data)
            settle_steps = round(
                float(self.config["simulation"]["settle_s"]) / self.model.opt.timestep
            )
            for _ in range(max(0, settle_steps)):
                mujoco.mj_step(self.model, self.data)
            self._current_seed = int(seed)
            self._latest_frames = self._render_all()
            return {"seed": int(seed), "placements": sampled}

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
        segmentation = self.renderer.render().astype(np.int32, copy=True)
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
        rng = np.random.default_rng(seed + 1)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=7)
        frequencies = rng.uniform(0.08, 0.18, size=7)
        home = np.asarray(self.config["robot"]["home_q"], dtype=np.float64)
        amplitude = np.asarray(self.config["robot"]["motion_amplitude"], dtype=np.float64)

        time_rows: list[float] = []
        qpos_rows: list[np.ndarray] = []
        qvel_rows: list[np.ndarray] = []
        ctrl_rows: list[np.ndarray] = []
        body_pos_rows: list[np.ndarray] = []
        body_quat_rows: list[np.ndarray] = []
        frame_records: list[dict[str, Any]] = []

        for frame_index, relative_time in enumerate(frame_times):
            target_time = start_time + float(relative_time)
            while self.data.time + self.model.opt.timestep * 0.5 < target_time:
                elapsed = float(self.data.time - start_time)
                target = home + amplitude * np.sin(2.0 * np.pi * frequencies * elapsed + phases)
                self.data.ctrl[self.arm_actuator_ids] = target
                mujoco.mj_step(self.model, self.data)

            modalities = self._render_all()
            self._latest_frames = modalities
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
            frame_records.append({"frame_id": frame_id, "time_s": time_rows[-1]})

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
        )
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
            "segmentation": {
                "raw_array": "*_segmentation.npy",
                "raw_array_channels": ["object_id", "object_type"],
                "geom_label_map": self._label_map(),
                "instance_array": "*_instance.npy",
                "instance_definition": "MuJoCo body id; -1 is background",
                "body_label_map": self._instance_label_map(),
            },
            "randomization": randomization,
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
