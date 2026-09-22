from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .robots import HAND_TYPES, HANDEDNESS, XHAND_LOWER, XHAND_OPEN_Q, XHAND_UPPER, hand_dof
from .teaware_assets import get_teaware_asset


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
        asset_id = str(obj.get("asset_id", "")).strip()
        if asset_id:
            try:
                get_teaware_asset(asset_id)
            except (KeyError, FileNotFoundError) as exc:
                raise ConfigError(f"{path}.asset_id is invalid: {exc}") from exc
            obj["asset_id"] = asset_id
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

    legacy_robot = config.pop("robot", None)
    robots = config.get("robots")
    if robots is None:
        legacy_robot = legacy_robot or {}
        robots = [
            {
                "id": "arm",
                "hand": "gripper",
                "handedness": "right",
                "base_position": [0.0, 0.0, 0.0],
                "base_yaw_deg": 0.0,
                **legacy_robot,
            }
        ]
        config["robots"] = robots
    if not isinstance(robots, list) or not robots:
        raise ConfigError("config.robots must be a non-empty list")

    robot_ids: set[str] = set()
    default_home = [0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0]
    default_amplitude = [0.05, 0.04, 0.05, 0.04, 0.04, 0.04, 0.05]
    for index, robot in enumerate(robots):
        path = f"robots[{index}]"
        if not isinstance(robot, dict):
            raise ConfigError(f"{path} must be a mapping")
        robot_id = str(robot.get("id", "")).strip()
        if not robot_id or not robot_id.replace("_", "").isalnum():
            raise ConfigError(f"{path}.id must contain only letters, numbers, and underscores")
        if robot_id in robot_ids:
            raise ConfigError(f"{path}.id must be unique")
        robot_ids.add(robot_id)
        hand = str(robot.get("hand", "gripper")).strip().lower()
        handedness = str(robot.get("handedness", "right")).strip().lower()
        if hand not in HAND_TYPES:
            raise ConfigError(f"{path}.hand must be one of {sorted(HAND_TYPES)}")
        if handedness not in HANDEDNESS:
            raise ConfigError(f"{path}.handedness must be one of {sorted(HANDEDNESS)}")
        robot["id"] = robot_id
        robot["hand"] = hand
        robot["handedness"] = handedness
        robot["base_position"] = _vec(
            robot.get("base_position", [0.0, 0.0, 0.0]), 3, f"{path}.base_position"
        )
        robot["base_yaw_deg"] = float(robot.get("base_yaw_deg", 0.0))
        robot["home_q"] = _vec(robot.get("home_q", default_home), 7, f"{path}.home_q")
        robot["motion_amplitude"] = _vec(
            robot.get("motion_amplitude", default_amplitude),
            7,
            f"{path}.motion_amplitude",
        )
        dof = hand_dof(hand)
        default_hand_home = list(XHAND_OPEN_Q) if hand == "xhand" else [0.0]
        robot["hand_home_q"] = _vec(
            robot.get("hand_home_q", default_hand_home), dof, f"{path}.hand_home_q"
        )
        robot["hand_motion_amplitude"] = _vec(
            robot.get("hand_motion_amplitude", [0.0] * dof),
            dof,
            f"{path}.hand_motion_amplitude",
        )
        if hand == "xhand":
            for joint_index, value in enumerate(robot["hand_home_q"]):
                if not XHAND_LOWER[joint_index] <= value <= XHAND_UPPER[joint_index]:
                    raise ConfigError(
                        f"{path}.hand_home_q[{joint_index}] is outside xHand limits"
                    )
    config["scene_profile"] = str(config.get("scene_profile", "custom"))
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
