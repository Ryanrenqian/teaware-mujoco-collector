from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def _require(mapping: dict[str, Any], key: str, expected: type, path: str) -> Any:
    value = mapping.get(key)
    if not isinstance(value, expected):
        raise ConfigError(f"{path}.{key} must be {expected.__name__}")
    return value


def _vec(value: Any, size: int, path: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ConfigError(f"{path} must contain {size} numbers")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path} must contain only numbers") from exc


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ConfigError("configuration root must be a mapping")

    simulation = _require(config, "simulation", dict, "config")
    timestep = float(simulation.get("timestep", 0.002))
    duration_s = float(simulation.get("duration_s", 1.0))
    capture_fps = float(simulation.get("capture_fps", 10.0))
    settle_s = float(simulation.get("settle_s", 0.4))
    if min(timestep, duration_s, capture_fps) <= 0 or settle_s < 0:
        raise ConfigError("simulation timing values must be positive (settle_s may be zero)")
    simulation.update(
        timestep=timestep,
        duration_s=duration_s,
        capture_fps=capture_fps,
        settle_s=settle_s,
    )

    renderer = _require(config, "renderer", dict, "config")
    renderer["width"] = int(renderer.get("width", 640))
    renderer["height"] = int(renderer.get("height", 480))
    if renderer["width"] < 16 or renderer["height"] < 16:
        raise ConfigError("renderer dimensions must be at least 16x16")

    table = _require(config, "table", dict, "config")
    table["size"] = _vec(table.get("size"), 3, "table.size")
    table["center"] = _vec(table.get("center"), 3, "table.center")
    table["rgba"] = _vec(table.get("rgba", [0.42, 0.31, 0.22, 1.0]), 4, "table.rgba")

    cameras = _require(config, "cameras", list, "config")
    if not cameras:
        raise ConfigError("at least one camera is required")
    camera_names: set[str] = set()
    for index, camera in enumerate(cameras):
        path = f"cameras[{index}]"
        if not isinstance(camera, dict):
            raise ConfigError(f"{path} must be a mapping")
        name = str(camera.get("name", "")).strip()
        if not name or name in camera_names:
            raise ConfigError(f"{path}.name must be unique and non-empty")
        camera_names.add(name)
        camera["name"] = name
        camera["position"] = _vec(camera.get("position"), 3, f"{path}.position")
        camera["target"] = _vec(camera.get("target"), 3, f"{path}.target")
        camera["fovy"] = float(camera.get("fovy", 48.0))
        if not 1.0 < camera["fovy"] < 179.0:
            raise ConfigError(f"{path}.fovy must be in (1, 179)")

    objects = _require(config, "objects", list, "config")
    if not objects:
        raise ConfigError("at least one teaware object is required")
    object_names: set[str] = set()
    allowed_presets = {"teapot", "teacup", "pitcher", "canister"}
    for index, obj in enumerate(objects):
        path = f"objects[{index}]"
        if not isinstance(obj, dict):
            raise ConfigError(f"{path} must be a mapping")
        name = str(obj.get("name", "")).strip()
        preset = str(obj.get("preset", "")).strip()
        if not name or name in object_names:
            raise ConfigError(f"{path}.name must be unique and non-empty")
        if preset not in allowed_presets:
            raise ConfigError(f"{path}.preset must be one of {sorted(allowed_presets)}")
        object_names.add(name)
        obj["name"] = name
        obj["preset"] = preset
        obj["rgba"] = _vec(obj.get("rgba", [0.75, 0.75, 0.72, 1.0]), 4, f"{path}.rgba")
        randomization = _require(obj, "randomization", dict, path)
        randomization["x"] = _vec(randomization.get("x"), 2, f"{path}.randomization.x")
        randomization["y"] = _vec(randomization.get("y"), 2, f"{path}.randomization.y")
        randomization["yaw_deg"] = _vec(
            randomization.get("yaw_deg", [-180.0, 180.0]),
            2,
            f"{path}.randomization.yaw_deg",
        )
        if (
            randomization["x"][0] > randomization["x"][1]
            or randomization["y"][0] > randomization["y"][1]
        ):
            raise ConfigError(f"{path}.randomization ranges must be ascending")

    randomization = config.setdefault("randomization", {})
    randomization["minimum_object_distance"] = float(
        randomization.get("minimum_object_distance", 0.12)
    )
    randomization["max_placement_attempts"] = int(randomization.get("max_placement_attempts", 200))
    if randomization["minimum_object_distance"] < 0:
        raise ConfigError("randomization.minimum_object_distance must be non-negative")

    robot = config.setdefault("robot", {})
    home = robot.get("home_q", [0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0])
    amplitude = robot.get("motion_amplitude", [0.05, 0.04, 0.05, 0.04, 0.04, 0.04, 0.05])
    robot["home_q"] = _vec(home, 7, "robot.home_q")
    robot["motion_amplitude"] = _vec(amplitude, 7, "robot.motion_amplitude")
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = validate_config(copy.deepcopy(raw))
    config["_source_path"] = str(config_path)
    return config


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(public_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
