from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PolicyObservation:
    episode_id: str
    step_index: int
    time_s: float
    task: str
    robot_ids: tuple[str, ...]
    hand_dof: np.ndarray
    arm_qpos: np.ndarray
    arm_qvel: np.ndarray
    hand_qpos: np.ndarray
    hand_qvel: np.ndarray
    tcp_position: np.ndarray
    tcp_quaternion: np.ndarray
    object_names: tuple[str, ...]
    object_position: np.ndarray
    object_quaternion: np.ndarray
    images_rgb: dict[str, np.ndarray] = field(repr=False)
    images_depth: dict[str, np.ndarray] = field(repr=False)

    @property
    def robot_count(self) -> int:
        return len(self.robot_ids)

    def to_wire(self, *, include_depth: bool = False, jpeg_quality: int = 90) -> dict[str, Any]:
        images: dict[str, dict[str, Any]] = {}
        for camera, rgb in self.images_rgb.items():
            buffer = io.BytesIO()
            Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
                buffer, format="JPEG", quality=int(jpeg_quality), subsampling=0
            )
            camera_payload: dict[str, Any] = {
                "rgb_jpeg_base64": base64.b64encode(buffer.getvalue()).decode("ascii")
            }
            if include_depth and camera in self.images_depth:
                depth_buffer = io.BytesIO()
                np.save(
                    depth_buffer,
                    np.asarray(self.images_depth[camera], dtype=np.float32),
                    allow_pickle=False,
                )
                camera_payload["depth_npy_base64"] = base64.b64encode(
                    depth_buffer.getvalue()
                ).decode("ascii")
            images[camera] = camera_payload
        return {
            "episode_id": self.episode_id,
            "step_index": int(self.step_index),
            "time_s": float(self.time_s),
            "task": self.task,
            "robot_ids": list(self.robot_ids),
            "hand_dof": self.hand_dof.astype(int).tolist(),
            "state": {
                "arm_qpos": self.arm_qpos.tolist(),
                "arm_qvel": self.arm_qvel.tolist(),
                "hand_qpos": self.hand_qpos.tolist(),
                "hand_qvel": self.hand_qvel.tolist(),
                "tcp_position": self.tcp_position.tolist(),
                "tcp_quaternion": self.tcp_quaternion.tolist(),
            },
            "objects": {
                "names": list(self.object_names),
                "position": self.object_position.tolist(),
                "quaternion": self.object_quaternion.tolist(),
            },
            "images": images,
        }


@dataclass(frozen=True)
class ActionChunk:
    arm_q_target: np.ndarray
    hand_q_target: np.ndarray
    dt_s: float
    action_mode: str = "joint_position"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arm = np.asarray(self.arm_q_target, dtype=np.float64)
        hand = np.asarray(self.hand_q_target, dtype=np.float64)
        if arm.ndim != 3 or arm.shape[2] != 7:
            raise ValueError(f"arm_q_target must have shape (H, R, 7), got {arm.shape}")
        if hand.ndim != 3 or hand.shape[:2] != arm.shape[:2] or hand.shape[2] != 12:
            raise ValueError(f"hand_q_target must have shape (H, R, 12), got {hand.shape}")
        if arm.shape[0] < 1 or arm.shape[1] < 1:
            raise ValueError("action chunk horizon and robot count must be positive")
        if not np.isfinite(arm).all() or not np.isfinite(hand).all():
            raise ValueError("action chunk contains non-finite targets")
        if not np.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("action chunk dt_s must be positive")
        if self.action_mode != "joint_position":
            raise ValueError(f"unsupported action_mode: {self.action_mode}")
        object.__setattr__(self, "arm_q_target", arm)
        object.__setattr__(self, "hand_q_target", hand)
        object.__setattr__(self, "dt_s", float(self.dt_s))

    @property
    def horizon(self) -> int:
        return int(self.arm_q_target.shape[0])

    @property
    def robot_count(self) -> int:
        return int(self.arm_q_target.shape[1])

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ActionChunk:
        action = payload.get("action", payload)
        return cls(
            arm_q_target=np.asarray(action["arm_q_target"], dtype=np.float64),
            hand_q_target=np.asarray(action["hand_q_target"], dtype=np.float64),
            dt_s=float(action["dt_s"]),
            action_mode=str(action.get("action_mode", "joint_position")),
            metadata=dict(action.get("metadata", payload.get("metadata", {}))),
        )


@runtime_checkable
class Policy(Protocol):
    def reset(self, seed: int, initial_observation: PolicyObservation) -> None: ...

    def act(self, observation: PolicyObservation) -> ActionChunk: ...

    def metadata(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class PolicyRunner:
    def __init__(self, policy: Policy, robot_count: int):
        self.policy = policy
        self.robot_count = int(robot_count)
        self.chunk: ActionChunk | None = None
        self.chunk_start_s = 0.0
        self.last_latency_ms = 0.0

    def reset(self, seed: int, observation: PolicyObservation) -> None:
        self.chunk = None
        self.chunk_start_s = float(observation.time_s)
        self.last_latency_ms = 0.0
        self.policy.reset(int(seed), observation)

    def submit(self, observation: PolicyObservation) -> ActionChunk:
        start = time.perf_counter()
        chunk = self.policy.act(observation)
        self.last_latency_ms = (time.perf_counter() - start) * 1000.0
        if not isinstance(chunk, ActionChunk):
            raise TypeError(f"policy returned {type(chunk).__name__}, expected ActionChunk")
        if chunk.robot_count != self.robot_count:
            raise ValueError(
                f"policy returned {chunk.robot_count} robots, expected {self.robot_count}"
            )
        self.chunk = chunk
        self.chunk_start_s = float(observation.time_s)
        return chunk

    def targets_at(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        if self.chunk is None:
            raise RuntimeError("policy has not produced an action chunk")
        elapsed = max(0.0, float(time_s) - self.chunk_start_s)
        index = min(int(elapsed / self.chunk.dt_s), self.chunk.horizon - 1)
        return self.chunk.arm_q_target[index], self.chunk.hand_q_target[index]
