from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def load_manifest(episode_dir: str | Path) -> dict[str, Any]:
    path = Path(episode_dir) / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _check_array(path: Path, shape: tuple[int | None, ...], errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing {path.name}")
        return
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        errors.append(f"cannot load {path.name}: {exc}")
        return
    if array.ndim != len(shape):
        errors.append(f"{path.name}: expected {len(shape)} dimensions, got {array.shape}")
        return
    for actual, expected in zip(array.shape, shape):
        if expected is not None and actual != expected:
            errors.append(f"{path.name}: expected shape {shape}, got {array.shape}")
            return
    if not np.isfinite(array).all():
        errors.append(f"{path.name}: contains non-finite values")


def validate_episode(episode_dir: str | Path) -> list[str]:
    root = Path(episode_dir)
    errors: list[str] = []
    try:
        manifest = load_manifest(root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"manifest.json: {exc}"]

    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {SCHEMA_VERSION}, got {manifest.get('schema_version')!r}"
        )
    frames = manifest.get("frames")
    cameras = manifest.get("cameras")
    if not isinstance(frames, list) or not frames:
        errors.append("manifest.frames must be a non-empty list")
        return errors
    if not isinstance(cameras, dict) or not cameras:
        errors.append("manifest.cameras must be a non-empty mapping")
        return errors

    height = int(manifest.get("renderer", {}).get("height", 0))
    width = int(manifest.get("renderer", {}).get("width", 0))
    if height <= 0 or width <= 0:
        errors.append("manifest.renderer dimensions are invalid")
        return errors

    for frame in frames:
        frame_id = str(frame.get("frame_id", ""))
        frame_dir = root / "frames" / frame_id
        if not frame_dir.is_dir():
            errors.append(f"missing frames/{frame_id}")
            continue
        for camera_name in cameras:
            prefix = frame_dir / camera_name
            for suffix in ("_rgb.jpg", "_depth_preview.png", "_segmentation_preview.png"):
                if not prefix.with_name(prefix.name + suffix).is_file():
                    errors.append(f"missing frames/{frame_id}/{camera_name}{suffix}")
            _check_array(
                prefix.with_name(prefix.name + "_depth.npy"),
                (height, width),
                errors,
            )
            _check_array(
                prefix.with_name(prefix.name + "_segmentation.npy"),
                (height, width, 2),
                errors,
            )
            _check_array(
                prefix.with_name(prefix.name + "_instance.npy"),
                (height, width),
                errors,
            )

    trajectory_path = root / "trajectory.npz"
    if not trajectory_path.is_file():
        errors.append("missing trajectory.npz")
    else:
        try:
            trajectory = np.load(trajectory_path, allow_pickle=False)
            frame_count = len(frames)
            for key in ("time", "qpos", "qvel", "ctrl", "body_position", "body_quaternion"):
                if key not in trajectory:
                    errors.append(f"trajectory.npz missing {key}")
                elif len(trajectory[key]) != frame_count:
                    errors.append(
                        f"trajectory.{key}: expected {frame_count} rows, got {len(trajectory[key])}"
                    )
        except (OSError, ValueError) as exc:
            errors.append(f"cannot load trajectory.npz: {exc}")
    return errors


def discover_episodes(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root).expanduser().resolve() / "episodes"
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.glob("episode_*")
        if path.is_dir() and (path / "manifest.json").is_file()
    )


def validate_dataset(dataset_root: str | Path) -> dict[str, list[str]]:
    return {episode.name: validate_episode(episode) for episode in discover_episodes(dataset_root)}
