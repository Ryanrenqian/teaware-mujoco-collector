from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from teaware_mujoco.cli import default_config_path
from teaware_mujoco.config import ConfigError, config_sha256, load_config


def test_default_config_is_valid() -> None:
    config = load_config(default_config_path())
    assert len(config["cameras"]) == 3
    assert {item["preset"] for item in config["objects"]} == {
        "teapot",
        "teacup",
        "pitcher",
        "canister",
    }
    assert len(config_sha256(config)) == 64


def test_duplicate_camera_name_is_rejected(tmp_path: Path) -> None:
    config = yaml.safe_load(default_config_path().read_text(encoding="utf-8"))
    config["cameras"][1]["name"] = config["cameras"][0]["name"]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="unique"):
        load_config(path)
