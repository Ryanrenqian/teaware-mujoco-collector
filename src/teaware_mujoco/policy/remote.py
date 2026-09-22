from __future__ import annotations

from typing import Any

import httpx

from .base import ActionChunk, PolicyObservation


class RemoteVLAPolicy:
    def __init__(
        self,
        url: str,
        *,
        timeout_s: float = 30.0,
        include_depth: bool = False,
        jpeg_quality: int = 90,
        control_hz: float = 10.0,
        action_horizon: int = 4,
        client: httpx.Client | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.include_depth = bool(include_depth)
        self.jpeg_quality = int(jpeg_quality)
        self.control_hz = float(control_hz)
        self.action_horizon = int(action_horizon)
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=float(timeout_s))
        self._server_metadata: dict[str, Any] = {}

    def reset(self, seed: int, initial_observation: PolicyObservation) -> None:
        del seed, initial_observation
        self._server_metadata = {}

    def act(self, observation: PolicyObservation) -> ActionChunk:
        payload = observation.to_wire(
            include_depth=self.include_depth,
            jpeg_quality=self.jpeg_quality,
        )
        payload["action_spec"] = {
            "action_mode": "joint_position",
            "control_hz": self.control_hz,
            "action_horizon": self.action_horizon,
            "arm_q_target_shape": [self.action_horizon, observation.robot_count, 7],
            "hand_q_target_shape": [self.action_horizon, observation.robot_count, 12],
        }
        response = self.client.post(
            f"{self.url}/v1/actions",
            json=payload,
        )
        response.raise_for_status()
        payload = response.json()
        self._server_metadata = dict(payload.get("model", {}))
        return ActionChunk.from_mapping(payload)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "remote_vla",
            "name": "remote_vla",
            "url": self.url,
            "include_depth": self.include_depth,
            "control_hz": self.control_hz,
            "action_horizon": self.action_horizon,
            "server_model": dict(self._server_metadata),
        }

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
