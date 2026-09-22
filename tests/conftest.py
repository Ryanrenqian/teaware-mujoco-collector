from __future__ import annotations

from pathlib import Path

import pytest

from teaware_mujoco.cli import default_config_path
from teaware_mujoco.config import load_config


@pytest.fixture
def tiny_config() -> dict:
    config = load_config(default_config_path())
    config["renderer"].update(width=64, height=48)
    config["simulation"].update(settle_s=0.01, duration_s=0.05, capture_fps=20.0)
    config["cameras"] = config["cameras"][:1]
    config["objects"] = config["objects"][:2]
    return config


@pytest.fixture
def collected_dataset(tmp_path: Path, tiny_config: dict) -> Path:
    from teaware_mujoco.collector import TeawareCollector

    root = tmp_path / "dataset"
    collector = TeawareCollector(tiny_config, root)
    try:
        collector.collect_episode(17)
    finally:
        collector.close()
    return root
