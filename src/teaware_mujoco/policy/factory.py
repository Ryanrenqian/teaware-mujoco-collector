from __future__ import annotations

from typing import Any

from .base import Policy
from .remote import RemoteVLAPolicy
from .scripted import ScriptedMotionPolicy


def build_policy(config: dict[str, Any]) -> Policy:
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
    raise ValueError(f"unsupported policy type: {policy['type']}")
