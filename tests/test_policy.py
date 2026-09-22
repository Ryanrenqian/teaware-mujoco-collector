from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from teaware_mujoco.collector import TeawareCollector
from teaware_mujoco.config import load_config
from teaware_mujoco.policy import (
    ActionChunk,
    LocalModelPolicy,
    PolicyObservation,
    PolicyRunner,
    RemoteVLAPolicy,
    ScriptedMotionPolicy,
    TimedTrajectoryPolicy,
)
from teaware_mujoco.policy.ik import pose_matrix
from teaware_mujoco.policy.mock_server import create_mock_vla_app
from teaware_mujoco.policy.tro import _as_pose

REPO_ROOT = Path(__file__).resolve().parents[1]


def _observation(robot_count: int = 1, *, time_s: float = 0.0) -> PolicyObservation:
    return PolicyObservation(
        episode_id="episode_000000",
        step_index=0,
        time_s=time_s,
        task="pick up the teapot",
        robot_ids=tuple(f"arm_{index}" for index in range(robot_count)),
        hand_dof=np.full(robot_count, 12, dtype=np.int32),
        arm_qpos=np.zeros((robot_count, 7)),
        arm_qvel=np.zeros((robot_count, 7)),
        hand_qpos=np.zeros((robot_count, 12)),
        hand_qvel=np.zeros((robot_count, 12)),
        tcp_position=np.zeros((robot_count, 3)),
        tcp_quaternion=np.tile([1.0, 0.0, 0.0, 0.0], (robot_count, 1)),
        object_names=("red_teapot",),
        object_position=np.zeros((1, 3)),
        object_quaternion=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        images_rgb={"front": np.zeros((12, 16, 3), dtype=np.uint8)},
        images_depth={"front": np.ones((12, 16), dtype=np.float32)},
    )


def test_action_chunk_rejects_wrong_or_nonfinite_shapes() -> None:
    with pytest.raises(ValueError, match="arm_q_target"):
        ActionChunk(np.zeros((1, 7)), np.zeros((1, 1, 12)), 0.1)
    with pytest.raises(ValueError, match="non-finite"):
        ActionChunk(np.full((1, 1, 7), np.nan), np.zeros((1, 1, 12)), 0.1)


def test_pose_matrix_applies_quaternion_rotation() -> None:
    angle = np.pi / 2.0
    transform = pose_matrix(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)]),
    )
    np.testing.assert_allclose(transform[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(transform[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-8)


def test_tro_http_pose_mapping_is_supported() -> None:
    transform = _as_pose({"position": [0.1, 0.2, 0.3], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]})
    np.testing.assert_allclose(transform[:3, :3], np.eye(3))
    np.testing.assert_allclose(transform[:3, 3], [0.1, 0.2, 0.3])


def test_scripted_motion_policy_returns_staged_action_chunk(tiny_config: dict) -> None:
    policy = ScriptedMotionPolicy(
        tiny_config["robots"], duration_s=1.0, control_hz=10.0, action_horizon=4
    )
    observation = _observation()
    policy.reset(7, observation)
    chunk = policy.act(observation)
    assert chunk.arm_q_target.shape == (4, 1, 7)
    assert chunk.hand_q_target.shape == (4, 1, 12)
    assert chunk.metadata["stage"] == "home"
    np.testing.assert_allclose(chunk.arm_q_target[0], [tiny_config["robots"][0]["home_q"]])


def test_local_model_policy_runs_through_policy_runner() -> None:
    observation = _observation()

    def predict(obs: PolicyObservation) -> dict:
        return {
            "arm_q_target": np.repeat(obs.arm_qpos[None], 2, axis=0),
            "hand_q_target": np.repeat(obs.hand_qpos[None], 2, axis=0),
            "dt_s": 0.05,
        }

    policy = LocalModelPolicy(predict, name="unit_test_model")
    runner = PolicyRunner(policy, robot_count=1)
    runner.reset(3, observation)
    chunk = runner.submit(observation)
    assert chunk.horizon == 2
    assert runner.targets_at(0.08)[0].shape == (1, 7)
    assert policy.metadata()["name"] == "unit_test_model"


def test_waic_timed_trajectory_adapter_interpolates_action_chunks() -> None:
    class WaicTimed:
        positions = np.asarray([[0.0] * 7, [1.0] * 7], dtype=np.float64)
        times = np.asarray([4.0, 5.0], dtype=np.float64)
        cmd_dt = 1.0

    policy = TimedTrajectoryPolicy.from_waic_timed_trajectory(
        WaicTimed(), control_hz=4.0, action_horizon=3
    )
    observation = _observation(time_s=0.25)
    policy.reset(0, observation)
    chunk = policy.act(observation)
    np.testing.assert_allclose(chunk.arm_q_target[:, 0, 0], [0.25, 0.5, 0.75])
    assert policy.metadata()["source_waypoints"] == 2


def test_remote_vla_policy_serializes_observation_and_parses_chunk() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["task"] == "pick up the teapot"
        assert payload["images"]["front"]["rgb_jpeg_base64"]
        assert payload["action_spec"]["arm_q_target_shape"] == [4, 1, 7]
        return httpx.Response(
            200,
            json={
                "action": {
                    "arm_q_target": np.zeros((3, 1, 7)).tolist(),
                    "hand_q_target": np.zeros((3, 1, 12)).tolist(),
                    "dt_s": 0.1,
                },
                "model": {"name": "test-vla"},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    policy = RemoteVLAPolicy("http://policy.test", client=client)
    chunk = policy.act(_observation())
    assert chunk.horizon == 3
    assert policy.metadata()["server_model"] == {"name": "test-vla"}
    client.close()


def test_mock_vla_server_holds_current_position() -> None:
    app = create_mock_vla_app(action_horizon=3, dt_s=0.2)
    payload = _observation().to_wire()
    with TestClient(app) as client:
        assert client.get("/health").json()["ok"] is True
        response = client.post("/v1/actions", json=payload)
    assert response.status_code == 200
    chunk = ActionChunk.from_mapping(response.json())
    assert chunk.arm_q_target.shape == (3, 1, 7)
    assert chunk.dt_s == 0.2


def test_collector_accepts_injected_local_policy(tmp_path: Path, tiny_config: dict) -> None:
    def predict(obs: PolicyObservation) -> ActionChunk:
        return ActionChunk(
            np.repeat(obs.arm_qpos[None], 2, axis=0),
            np.repeat(obs.hand_qpos[None], 2, axis=0),
            0.05,
            metadata={"stage": "hold"},
        )

    policy = LocalModelPolicy(predict, name="collector_test_model")
    collector = TeawareCollector(tiny_config, tmp_path / "dataset", policy=policy)
    try:
        episode = collector.collect_episode(5)
    finally:
        collector.close()

    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    trajectory = np.load(episode / "trajectory.npz", allow_pickle=False)
    assert manifest["policy"]["type"] == "local_vla"
    assert manifest["policy"]["name"] == "collector_test_model"
    assert trajectory["policy_arm_q_target"].shape == (2, 1, 7)
    assert trajectory["policy_hand_q_target"].shape == (2, 1, 12)
    assert trajectory["policy_stage"].tolist() == ["hold", "hold"]
    assert np.all(trajectory["policy_latency_ms"] >= 0.0)


def test_policy_failure_rolls_back_temporary_episode(tmp_path: Path, tiny_config: dict) -> None:
    def fail(_observation: PolicyObservation) -> ActionChunk:
        raise RuntimeError("inference failed")

    policy = LocalModelPolicy(fail, name="failing_model")
    root = tmp_path / "dataset"
    collector = TeawareCollector(tiny_config, root, policy=policy)
    try:
        with pytest.raises(RuntimeError, match="inference failed"):
            collector.collect_episode(5)
    finally:
        collector.close()

    assert list((root / "episodes").iterdir()) == []
    assert not (root / "dataset.jsonl").exists()


def test_tro_mock_policy_plans_and_lifts_target_object(tmp_path: Path) -> None:
    config = load_config(REPO_ROOT / "configs" / "tro_xhand_mock.yaml")
    config["renderer"].update(width=64, height=48)
    config["cameras"] = config["cameras"][:1]
    config["simulation"].update(settle_s=0.01, duration_s=1.9, capture_fps=10.0)
    config["policy"]["tro"]["stage_durations_s"] = {
        key: 0.3 for key in ("pregrasp", "grasp", "close", "lift", "return", "release")
    }
    config["policy"]["tro"]["grasp_constraint"]["max_distance_m"] = 0.30
    collector = TeawareCollector(config, tmp_path / "tro_dataset")
    try:
        episode = collector.collect_episode(7)
    finally:
        collector.close()

    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    with np.load(episode / "trajectory.npz", allow_pickle=False) as trajectory:
        stages = list(dict.fromkeys(trajectory["policy_stage"].tolist()))
        target_z = trajectory["body_position"][:, 0, 2]
        assert target_z.max() > target_z[0] + 0.04
    assert manifest["policy"]["type"] == "tro_grasp"
    assert manifest["policy"]["tro"]["backend"] == "centroid_mock"
    assert stages == ["pregrasp", "grasp", "close", "lift", "return", "release"]
