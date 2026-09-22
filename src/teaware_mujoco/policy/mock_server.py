from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException

from .. import __version__


def create_mock_vla_app(*, action_horizon: int = 4, dt_s: float = 0.1) -> FastAPI:
    horizon = int(action_horizon)
    action_dt = float(dt_s)
    app = FastAPI(title="Teaware Mock VLA Server", version=__version__)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "model": "hold_position_mock_v1"}

    @app.post("/v1/actions")
    def actions(observation: dict[str, Any]) -> dict[str, Any]:
        try:
            state = observation["state"]
            arm = np.asarray(state["arm_qpos"], dtype=np.float64)
            hand = np.asarray(state["hand_qpos"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"invalid observation: {exc}") from exc
        if arm.ndim != 2 or arm.shape[1] != 7 or hand.shape != (arm.shape[0], 12):
            raise HTTPException(status_code=422, detail="invalid arm/hand state shape")
        action_spec = observation.get("action_spec", {})
        requested_horizon = int(action_spec.get("action_horizon", horizon))
        requested_hz = float(action_spec.get("control_hz", 1.0 / action_dt))
        if not 1 <= requested_horizon <= 128 or requested_hz <= 0:
            raise HTTPException(status_code=422, detail="invalid action_spec")
        return {
            "action": {
                "arm_q_target": np.repeat(arm[None], requested_horizon, axis=0).tolist(),
                "hand_q_target": np.repeat(hand[None], requested_horizon, axis=0).tolist(),
                "dt_s": 1.0 / requested_hz,
                "action_mode": "joint_position",
                "metadata": {"stage": "hold"},
            },
            "model": {"name": "hold_position_mock_v1", "version": __version__},
        }

    return app
