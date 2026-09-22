from __future__ import annotations

from typing import Any

import numpy as np

from .base import ActionChunk, PolicyObservation


class ScriptedMotionPolicy:
    """Deterministic staged joint-space baseline for collection and policy plumbing."""

    def __init__(
        self,
        robots: list[dict[str, Any]],
        *,
        duration_s: float,
        control_hz: float,
        action_horizon: int,
    ) -> None:
        self.duration_s = float(duration_s)
        self.dt_s = 1.0 / float(control_hz)
        self.action_horizon = int(action_horizon)
        self.arm_home = np.asarray([robot["home_q"] for robot in robots], dtype=np.float64)
        self.arm_delta = np.asarray(
            [robot["motion_amplitude"] for robot in robots], dtype=np.float64
        )
        self.hand_home = np.zeros((len(robots), 12), dtype=np.float64)
        self.hand_delta = np.zeros((len(robots), 12), dtype=np.float64)
        for index, robot in enumerate(robots):
            dof = len(robot["hand_home_q"])
            self.hand_home[index, :dof] = robot["hand_home_q"]
            self.hand_delta[index, :dof] = robot["hand_motion_amplitude"]

    def reset(self, seed: int, initial_observation: PolicyObservation) -> None:
        del seed, initial_observation

    @staticmethod
    def _interpolate_keyframes(
        progress: np.ndarray, keyframes: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        output = np.empty((len(progress), *values.shape[1:]), dtype=np.float64)
        for index, value in enumerate(progress):
            right = min(int(np.searchsorted(keyframes, value, side="right")), len(keyframes) - 1)
            left = max(0, right - 1)
            span = float(keyframes[right] - keyframes[left])
            alpha = 0.0 if span <= 0 else float(value - keyframes[left]) / span
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            output[index] = (1.0 - alpha) * values[left] + alpha * values[right]
        return output

    def act(self, observation: PolicyObservation) -> ActionChunk:
        times = observation.time_s + np.arange(self.action_horizon, dtype=np.float64) * self.dt_s
        progress = np.clip(times / max(self.duration_s, self.dt_s), 0.0, 1.0)
        keyframes = np.asarray([0.0, 0.25, 0.5, 0.72, 1.0], dtype=np.float64)
        arm_values = np.stack(
            [
                self.arm_home,
                self.arm_home + self.arm_delta,
                self.arm_home + self.arm_delta,
                self.arm_home - 0.5 * self.arm_delta,
                self.arm_home,
            ]
        )
        hand_values = np.stack(
            [
                self.hand_home,
                self.hand_home,
                self.hand_home + self.hand_delta,
                self.hand_home + self.hand_delta,
                self.hand_home,
            ]
        )
        stage_index = min(
            max(int(np.searchsorted(keyframes, progress[0], side="right")) - 1, 0), 4
        )
        stages = ("home", "approach", "close", "lift", "return")
        return ActionChunk(
            arm_q_target=self._interpolate_keyframes(progress, keyframes, arm_values),
            hand_q_target=self._interpolate_keyframes(progress, keyframes, hand_values),
            dt_s=self.dt_s,
            metadata={"stage": stages[stage_index]},
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "scripted_motion",
            "name": "staged_joint_motion_v1",
            "control_hz": 1.0 / self.dt_s,
            "action_horizon": self.action_horizon,
        }

    def close(self) -> None:
        return None
