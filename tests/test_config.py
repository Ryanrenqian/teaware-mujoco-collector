from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from teaware_mujoco.cli import default_config_path
from teaware_mujoco.config import ConfigError, config_sha256, load_config
from teaware_mujoco.teaware_assets import available_asset_ids, get_teaware_asset

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    assert config["policy"]["type"] == "scripted_motion"
    assert [item["asset_id"] for item in config["objects"]] == [
        "teapot_porcelain_red__object_000",
        "teapot_porcelain_red__object_001",
        "teapot_porcelain_red__object_002",
        "teapot_porcelain_red__object_003",
    ]


def test_teaware_catalog_contains_complete_mesh_assets() -> None:
    asset_ids = available_asset_ids()
    assert len(asset_ids) == 40
    for asset_id in asset_ids:
        asset = get_teaware_asset(asset_id)
        assert len(asset["collisions"]) == 16
        assert len(asset["extents_m"]) == 3


def test_duplicate_camera_name_is_rejected(tmp_path: Path) -> None:
    config = yaml.safe_load(default_config_path().read_text(encoding="utf-8"))
    config["cameras"][1]["name"] = config["cameras"][0]["name"]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="unique"):
        load_config(path)


def test_unknown_teaware_asset_is_rejected(tmp_path: Path) -> None:
    config = yaml.safe_load(default_config_path().read_text(encoding="utf-8"))
    config["objects"][0]["asset_id"] = "missing_asset"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="asset_id"):
        load_config(path)


def test_remote_vla_policy_requires_http_url(tmp_path: Path) -> None:
    config = yaml.safe_load(default_config_path().read_text(encoding="utf-8"))
    config["policy"] = {"type": "remote_vla", "url": "not-a-url"}
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="http"):
        load_config(path)


def test_tro_configs_define_independent_local_and_mock_backends() -> None:
    local = load_config(REPO_ROOT / "configs" / "tro_xhand.yaml")
    mock = load_config(REPO_ROOT / "configs" / "tro_xhand_mock.yaml")
    assert local["policy"]["type"] == "tro_grasp"
    assert local["policy"]["tro"]["backend"] == "local"
    assert local["policy"]["tro"]["root"] == "${TRO_ROOT}"
    assert mock["policy"]["tro"]["backend"] == "centroid_mock"
    assert mock["policy"]["tro"]["grasp_constraint"]["enabled"] is True


def test_tro_grasp_requires_duration_for_all_stages(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "tro_xhand_mock.yaml").read_text(encoding="utf-8")
    )
    config["simulation"]["duration_s"] = 1.0
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="complete TRO grasp"):
        load_config(path)


@pytest.mark.parametrize(
    ("name", "profile", "hands"),
    [
        ("single_gripper.yaml", "single_gripper", ["gripper"]),
        ("single_xhand.yaml", "single_xhand", ["xhand"]),
        ("dual_xhand.yaml", "dual_xhand", ["xhand", "xhand"]),
    ],
)
def test_scenario_configs(name: str, profile: str, hands: list[str]) -> None:
    config = load_config(REPO_ROOT / "configs" / name)
    assert config["scene_profile"] == profile
    assert [robot["hand"] for robot in config["robots"]] == hands
    assert len({robot["id"] for robot in config["robots"]}) == len(hands)
    assert "return stages" in config["policy"]["task"]
