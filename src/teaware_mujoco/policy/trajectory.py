from __future__ import annotations

from typing import Any

import numpy as np

from .base import ActionChunk, PolicyObservation


def _arm_trajectory(array: Any) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] % 7 == 0:
        values = values.reshape(values.shape[0], values.shape[1] // 7, 7)
    if values.ndim != 3 or values.shape[2] != 7 or values.shape[0] < 1:
        raise ValueError(f"arm trajectory must have shape (N, R, 7), got {values.shape}")
    return values


def _hand_trajectory(array: Any, robot_count: int) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == robot_count * 12:
        values = values.reshape(values.shape[0], robot_count, 12)
    if values.ndim != 3 or values.shape[1:] != (robot_count, 12):
        raise ValueError(f"hand trajectory must have shape (N, {robot_count}, 12), got {values.shape}")
    return values


class TimedTrajectoryPolicy:
    """Adapter for planned joint trajectories, including waic-demo4 TimedTrajectory."""

    def __init__(
        self,
        arm_positions: Any,
        *,
        times: Any,
        hand_positions: Any | None = None,
        control_hz: float = 10.0,
        action_horizon: int = 4,
        name: str = "timed_trajectory",
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.arm_positions = _arm_trajectory(arm_positions)
        self.times = np.asarray(times, dtype=np.float64).reshape(-1)
        if len(self.times) != len(self.arm_positions):
            raise ValueError("trajectory times and positions must have the same length")
        if not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError("trajectory times must be finite and strictly increasing")
        self.times = self.times - self.times[0]
        self.hand_positions = (
            None
            if hand_positions is None
            else _hand_trajectory(hand_positions, self.arm_positions.shape[1])
        )
        if self.hand_positions is not None and len(self.hand_positions) != len(self.times):
            raise ValueError("hand trajectory times and positions must have the same length")
        if control_hz <= 0 or action_horizon <= 0:
            raise ValueError("control_hz and action_horizon must be positive")
        self.dt_s = 1.0 / float(control_hz)
        self.action_horizon = int(action_horizon)
        self.name = str(name)
        self.provenance = dict(provenance or {})

    @classmethod
    def from_waic_timed_trajectory(
        cls,
        timed: Any,
        *,
        hand_positions: Any | None = None,
        control_hz: float = 10.0,
        action_horizon: int = 4,
        name: str = "waic_timed_trajectory",
        provenance: dict[str, Any] | None = None,
    ) -> TimedTrajectoryPolicy:
        positions = np.asarray(timed.positions, dtype=np.float64)
        raw_times = getattr(timed, "times", None)
        if raw_times is None or len(np.asarray(raw_times).reshape(-1)) != len(positions):
            raw_times = np.arange(len(positions), dtype=np.float64) * float(timed.cmd_dt)
        return cls(
            positions,
            times=raw_times,
            hand_positions=hand_positions,
            control_hz=control_hz,
            action_horizon=action_horizon,
            name=name,
            provenance=provenance,
        )

    def reset(self, seed: int, initial_observation: PolicyObservation) -> None:
        del seed
        if initial_observation.robot_count != self.arm_positions.shape[1]:
            raise ValueError(
                f"trajectory has {self.arm_positions.shape[1]} robots, "
                f"observation has {initial_observation.robot_count}"
            )

    def _sample(self, values: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
        output = np.empty((len(sample_times), *values.shape[1:]), dtype=np.float64)
        for robot in range(values.shape[1]):
            for joint in range(values.shape[2]):
                output[:, robot, joint] = np.interp(
                    sample_times,
                    self.times,
                    values[:, robot, joint],
                )
        return output

    def act(self, observation: PolicyObservation) -> ActionChunk:
        sample_times = observation.time_s + np.arange(self.action_horizon) * self.dt_s
        arm = self._sample(self.arm_positions, sample_times)
        hand = (
            np.repeat(observation.hand_qpos[None], self.action_horizon, axis=0)
            if self.hand_positions is None
            else self._sample(self.hand_positions, sample_times)
        )
        return ActionChunk(
            arm_q_target=arm,
            hand_q_target=hand,
            dt_s=self.dt_s,
            metadata={"stage": "trajectory"},
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "timed_trajectory",
            "name": self.name,
            "control_hz": 1.0 / self.dt_s,
            "action_horizon": self.action_horizon,
            "source_waypoints": len(self.times),
            "duration_s": float(self.times[-1]),
            **self.provenance,
        }

    def close(self) -> None:
        return None
