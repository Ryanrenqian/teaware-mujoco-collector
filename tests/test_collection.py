from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from teaware_mujoco.collector import _segmentation_from_idcolor
from teaware_mujoco.schema import discover_episodes, validate_dataset, validate_episode

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_segmentation_idcolor_ignores_out_of_range_pixels() -> None:
    pixels = np.array([[[1, 0, 0], [1, 4, 0]]], dtype=np.uint8)
    scene = SimpleNamespace(
        ngeom=1,
        geoms=[SimpleNamespace(segid=0, objid=7, objtype=5)],
    )
    segmentation = _segmentation_from_idcolor(pixels, scene)

    assert segmentation.tolist() == [[[7, 5], [-1, -1]]]


def test_episode_contains_synchronized_modalities(collected_dataset: Path) -> None:
    episodes = discover_episodes(collected_dataset)
    assert len(episodes) == 1
    episode = episodes[0]
    assert validate_episode(episode) == []
    assert validate_dataset(collected_dataset) == {"episode_000000": []}

    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 17
    assert manifest["renderer"] == {"width": 64, "height": 48}
    assert len(manifest["frames"]) == 2
    assert list(manifest["cameras"]) == ["front_left"]

    depth = np.load(episode / "frames/000000/front_left_depth.npy", allow_pickle=False)
    segmentation = np.load(
        episode / "frames/000000/front_left_segmentation.npy", allow_pickle=False
    )
    instance = np.load(episode / "frames/000000/front_left_instance.npy", allow_pickle=False)
    trajectory = np.load(episode / "trajectory.npz", allow_pickle=False)
    assert depth.shape == (48, 64)
    assert depth.dtype == np.float32
    assert segmentation.shape == (48, 64, 2)
    assert instance.shape == (48, 64)
    assert instance.dtype == np.int32
    assert len(trajectory["time"]) == len(manifest["frames"])
    assert trajectory["body_position"].shape == (2, 2, 3)
    assert trajectory["policy_arm_q_target"].shape == (2, 1, 7)
    assert trajectory["policy_hand_q_target"].shape == (2, 1, 12)
    assert manifest["policy"]["type"] == "scripted_motion"


def test_dataset_index_is_committed_after_episode(collected_dataset: Path) -> None:
    rows = (collected_dataset / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["episode_id"] == "episode_000000"


def test_dual_xhand_episode_has_grouped_robot_state(tmp_path: Path) -> None:
    from teaware_mujoco.collector import TeawareCollector
    from teaware_mujoco.config import load_config

    config = load_config(REPO_ROOT / "configs/dual_xhand.yaml")
    config["renderer"].update(width=64, height=48)
    config["simulation"].update(settle_s=0.01, duration_s=0.05, capture_fps=20.0)
    config["cameras"] = config["cameras"][:1]
    config["objects"] = config["objects"][:2]
    root = tmp_path / "dual_dataset"
    collector = TeawareCollector(config, root)
    try:
        episode = collector.collect_episode(41)
    finally:
        collector.close()

    assert validate_episode(episode) == []
    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scene_profile"] == "dual_xhand"
    assert [robot["id"] for robot in manifest["robots"]] == ["left_arm", "right_arm"]
    trajectory = np.load(episode / "trajectory.npz", allow_pickle=False)
    assert trajectory["arm_qpos"].shape == (2, 2, 7)
    assert trajectory["hand_qpos"].shape == (2, 2, 12)
    assert trajectory["tcp_position"].shape == (2, 2, 3)
    assert trajectory["policy_arm_q_target"].shape == (2, 2, 7)
    assert trajectory["policy_hand_q_target"].shape == (2, 2, 12)
    assert trajectory["robot_ids"].tolist() == ["left_arm", "right_arm"]
