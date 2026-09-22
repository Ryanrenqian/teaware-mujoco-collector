from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from teaware_mujoco.schema import discover_episodes, validate_dataset, validate_episode


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


def test_dataset_index_is_committed_after_episode(collected_dataset: Path) -> None:
    rows = (collected_dataset / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["episode_id"] == "episode_000000"
