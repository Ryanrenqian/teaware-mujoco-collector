from __future__ import annotations

from typing import Any

import mujoco

from .base import Policy
from .remote import RemoteVLAPolicy
from .scripted import ScriptedMotionPolicy
from .tro import TROGraspPolicy


def build_policy(
    config: dict[str, Any],
    *,
    model: mujoco.MjModel | None = None,
    robots: list[dict[str, Any]] | None = None,
) -> Policy:
    policy = config["policy"]
    if policy["type"] == "scripted_motion":
        return ScriptedMotionPolicy(
            config["robots"],
            duration_s=config["simulation"]["duration_s"],
            control_hz=policy["control_hz"],
            action_horizon=policy["action_horizon"],
        )
    if policy["type"] == "remote_vla":
        return RemoteVLAPolicy(
            policy["url"],
            timeout_s=policy["timeout_s"],
            include_depth=policy["include_depth"],
            jpeg_quality=policy["jpeg_quality"],
            control_hz=policy["control_hz"],
            action_horizon=policy["action_horizon"],
        )
    if policy["type"] == "tro_grasp":
        if model is None or robots is None:
            raise ValueError("tro_grasp policy requires a MuJoCo model and robot runtime")
        return TROGraspPolicy(config, model, robots)
    raise ValueError(f"unsupported policy type: {policy['type']}")
