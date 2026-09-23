from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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

    policy = config.setdefault("policy", {})
    if not isinstance(policy, dict):
        raise ConfigError("config.policy must be a mapping")
    policy_type = str(policy.get("type", "scripted_motion")).strip().lower()
    if policy_type not in {"scripted_motion", "remote_vla", "tro_grasp"}:
        raise ConfigError("config.policy.type must be scripted_motion, remote_vla, or tro_grasp")
    task = str(policy.get("task", "Move the robot through a staged collection trajectory.")).strip()
    if not task:
        raise ConfigError("config.policy.task must be non-empty")
    policy.update(
        type=policy_type,
        task=task,
        control_hz=float(policy.get("control_hz", capture_fps)),
        action_horizon=int(policy.get("action_horizon", 4)),
    )
    if policy["control_hz"] <= 0 or policy["action_horizon"] <= 0:
        raise ConfigError("config.policy control_hz and action_horizon must be positive")
    if policy_type == "remote_vla":
        url = str(policy.get("url", "")).strip().rstrip("/")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigError("config.policy.url must be an http(s) URL for remote_vla")
        policy.update(
            url=url,
            timeout_s=float(policy.get("timeout_s", 30.0)),
            include_depth=bool(policy.get("include_depth", False)),
            jpeg_quality=int(policy.get("jpeg_quality", 90)),
        )
        if policy["timeout_s"] <= 0 or not 1 <= policy["jpeg_quality"] <= 100:
            raise ConfigError("remote_vla timeout_s and jpeg_quality are invalid")
    elif policy_type == "tro_grasp":
        policy["robot_id"] = str(policy.get("robot_id", "")).strip()
        policy["target_object"] = str(policy.get("target_object", "")).strip()
        if not policy["robot_id"] or not policy["target_object"]:
            raise ConfigError("tro_grasp requires policy.robot_id and policy.target_object")
        tro = _require(policy, "tro", dict, "config.policy")
        backend = str(tro.get("backend", "local")).strip().lower()
        if backend not in {"local", "http", "centroid_mock"}:
            raise ConfigError("config.policy.tro.backend must be local, http, or centroid_mock")
        tro.update(
            backend=backend,
            hand_type=str(tro.get("hand_type", "xhand")).strip(),
            num_candidates=int(tro.get("num_candidates", 8)),
            object_points=int(tro.get("object_points", 2048)),
            environment_points=int(tro.get("environment_points", 4096)),
            root_to_tcp_position=_vec(
                tro.get("root_to_tcp_position", [0.0, 0.0, -0.065]),
                3,
                "policy.tro.root_to_tcp_position",
            ),
            root_to_tcp_quaternion_wxyz=_vec(
                tro.get("root_to_tcp_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]),
                4,
                "policy.tro.root_to_tcp_quaternion_wxyz",
            ),
            approach_axis=_vec(
                tro.get("approach_axis", [0.0, -1.0, 0.0]),
                3,
                "policy.tro.approach_axis",
            ),
            pregrasp_offset_m=float(tro.get("pregrasp_offset_m", 0.10)),
            lift_m=float(tro.get("lift_m", 0.12)),
            fallback_hand_q=_vec(
                tro.get(
                    "fallback_hand_q",
                    [1.35, -0.55, 0.55, 0.0, 1.25, 1.10, 1.25, 1.10, 1.2, 1.05, 1.15, 1.0],
                ),
                12,
                "policy.tro.fallback_hand_q",
            ),
        )
        if min(tro["num_candidates"], tro["object_points"], tro["environment_points"]) <= 0:
            raise ConfigError("TRO candidate and point counts must be positive")
        if tro["pregrasp_offset_m"] <= 0 or tro["lift_m"] <= 0:
            raise ConfigError("TRO pregrasp_offset_m and lift_m must be positive")
        grasp_constraint = tro.setdefault("grasp_constraint", {})
        if not isinstance(grasp_constraint, dict):
            raise ConfigError("config.policy.tro.grasp_constraint must be a mapping")
        grasp_constraint.update(
            enabled=bool(grasp_constraint.get("enabled", False)),
            max_distance_m=float(grasp_constraint.get("max_distance_m", 0.16)),
            min_finger_contacts=int(grasp_constraint.get("min_finger_contacts", 2)),
            min_closure_norm_rad=float(grasp_constraint.get("min_closure_norm_rad", 0.35)),
        )
        if grasp_constraint["max_distance_m"] <= 0:
            raise ConfigError("policy.tro.grasp_constraint.max_distance_m must be positive")
        if grasp_constraint["min_finger_contacts"] < 0:
            raise ConfigError("policy.tro.grasp_constraint.min_finger_contacts cannot be negative")
        if grasp_constraint["min_closure_norm_rad"] < 0:
            raise ConfigError("policy.tro.grasp_constraint.min_closure_norm_rad cannot be negative")
        if grasp_constraint["enabled"] and grasp_constraint["min_finger_contacts"] < 1:
            raise ConfigError(
                "enabled grasp_constraint requires at least one finger contact"
            )
        if grasp_constraint["enabled"] and grasp_constraint["min_closure_norm_rad"] <= 0:
            raise ConfigError("enabled grasp_constraint requires positive hand closure")
        durations = tro.setdefault("stage_durations_s", {})
        if not isinstance(durations, dict):
            raise ConfigError("config.policy.tro.stage_durations_s must be a mapping")
        for key, default in {
            "pregrasp": 1.5,
            "grasp": 1.0,
            "close": 0.7,
            "lift": 1.0,
            "return": 1.5,
            "release": 0.5,
        }.items():
            durations[key] = float(durations.get(key, default))
            if durations[key] <= 0:
                raise ConfigError(f"policy.tro.stage_durations_s.{key} must be positive")
        if duration_s + 1e-9 < sum(durations.values()):
            raise ConfigError(
                "simulation.duration_s must cover the complete TRO grasp stage durations"
            )
        if backend == "local":
            for key in ("root", "config", "checkpoint"):
                tro[key] = str(tro.get(key, "")).strip()
                if not tro[key]:
                    raise ConfigError(f"local TRO requires config.policy.tro.{key}")
            tro["device"] = str(tro.get("device", "")).strip() or None
            tro["inference_steps"] = int(tro.get("inference_steps", 20))
            tro["noise_lambda"] = float(tro.get("noise_lambda", 0.2))
            tro["root_link_name"] = str(tro.get("root_link_name", "")).strip() or None
        elif backend == "http":
            url = str(tro.get("url", "")).strip().rstrip("/")
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ConfigError("HTTP TRO requires config.policy.tro.url")
            tro["url"] = url
            tro["timeout_s"] = float(tro.get("timeout_s", 120.0))
            if tro["timeout_s"] <= 0:
                raise ConfigError("config.policy.tro.timeout_s must be positive")

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
    if policy_type == "tro_grasp":
        if policy["robot_id"] not in robot_ids:
            raise ConfigError(f"tro_grasp robot_id {policy['robot_id']!r} is not configured")
        selected_robot = next(robot for robot in robots if robot["id"] == policy["robot_id"])
        if selected_robot["hand"] != "xhand":
            raise ConfigError("tro_grasp currently requires an xhand robot")
        if policy["target_object"] not in object_names:
            raise ConfigError(
                f"tro_grasp target_object {policy['target_object']!r} is not configured"
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
