from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from teaware_mujoco.collector import TeawareCollector
from teaware_mujoco.config import ConfigError, load_config, validate_config
from teaware_mujoco.placement import PlacementError, TrayPlacement
from teaware_mujoco.scene import write_scene_xml
from teaware_mujoco.teaware_assets import get_teaware_asset, teaware_asset_dir

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dual_geometry(tmp_path_factory):
    config = load_config(ROOT / "configs/dual_xhand.yaml")
    xml = write_scene_xml(config, tmp_path_factory.mktemp("placement") / "scene.xml")
    model = mujoco.MjModel.from_xml_path(str(xml))
    # Independent source OBJ vertices, before MuJoCo recentres the compiled meshes.
    source_vertices = []
    for spec in config["objects"]:
        asset = get_teaware_asset(spec["asset_id"])
        points = []
        for relative in [asset["visual"], *asset["collisions"]]:
            for line in (teaware_asset_dir() / relative).read_text().splitlines():
                if line.startswith("v "):
                    points.append([float(value) for value in line.split()[1:4]])
        source_vertices.append(np.array(points))
    return config, model, source_vertices


def test_sampling_keeps_rotated_visual_and_collision_meshes_inside_tray(dual_geometry):
    config, model, vertices = dual_geometry
    placement = TrayPlacement(model, config)
    for seed in range(100):
        poses = placement.sample(np.random.default_rng(seed))
        footprints = []
        for spec, points, (x, y, yaw) in zip(config["objects"], vertices, poses):
            assert spec["randomization"]["x"][0] <= x <= spec["randomization"]["x"][1]
            assert spec["randomization"]["y"][0] <= y <= spec["randomization"]["y"][1]
            angle = np.deg2rad(yaw)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            world = points[:, :2] @ rotation.T + [x, y]
            low, high = world.min(axis=0), world.max(axis=0)
            # Green tray [0.19,0.73] x [-0.23,0.23], inset by 0.02 m.
            assert np.all(low >= np.array([0.21, -0.21]) - 1e-7)
            assert np.all(high <= np.array([0.71, 0.21]) + 1e-7)
            for old_low, old_high in footprints:
                separation = np.maximum(0, np.maximum(old_low - high, low - old_high))
                assert np.linalg.norm(separation) >= 0.01 - 1e-7
            footprints.append((low, high))
        for i, pose in enumerate(poses):
            for other in poses[:i]:
                assert np.linalg.norm(np.array(pose[:2]) - other[:2]) >= 0.13
    assert placement.sample(np.random.default_rng(12)) == placement.sample(
        np.random.default_rng(12)
    )


def test_rotation_and_margin_determine_feasible_centres(dual_geometry):
    original, model, _ = dual_geometry
    config = copy.deepcopy(original)
    config["objects"] = config["objects"][:1]
    ranges = config["objects"][0]["randomization"]
    ranges.update(x=[0.26, 0.26], y=[0, 0], yaw_deg=[0, 0])
    config["randomization"]["max_placement_attempts"] = 3
    placement = TrayPlacement(model, config)
    with pytest.raises(PlacementError, match="red_teapot.*edge_margin_m"):
        placement.sample(np.random.default_rng(0))
    ranges["yaw_deg"] = [90, 90]
    assert placement.sample(np.random.default_rng(0)) == [(0.26, 0.0, 90.0)]


@pytest.fixture
def dual_collector(tmp_path):
    config = load_config(ROOT / "configs/dual_xhand.yaml")
    config["renderer"].update(width=64, height=48)
    config["cameras"] = config["cameras"][:1]
    collector = TeawareCollector(config, tmp_path / "dataset")
    try:
        yield collector
    finally:
        collector.close()


def test_settled_resets_preserve_margin_support_and_seed(dual_collector, dual_geometry):
    collector = dual_collector
    _, _, vertices = dual_geometry
    first = collector.randomize(0)
    state = collector.data.qpos.copy()
    for seed in range(20):
        report = collector.randomize(seed)
        assert report["settled"]["valid"]
        for spec, points in zip(collector.object_specs, vertices):
            body_id = mujoco.mj_name2id(collector.model, mujoco.mjtObj.mjOBJ_BODY, spec["name"])
            rotation = collector.data.xmat[body_id].reshape(3, 3)
            world = points @ rotation.T + collector.data.xpos[body_id]
            assert np.all(world[:, :2].min(0) >= np.array([0.21, -0.21]) - 1e-7)
            assert np.all(world[:, :2].max(0) <= np.array([0.71, 0.21]) + 1e-7)
            contacts = []
            for contact in collector.data.contact[: collector.data.ncon]:
                if contact.efc_address < 0:
                    continue
                for own, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                    if collector.model.geom_bodyid[own] == body_id:
                        contacts.append(
                            mujoco.mj_id2name(collector.model, mujoco.mjtObj.mjOBJ_GEOM, other)
                        )
            assert set(contacts) == {"tea_tray"}
    assert collector.randomize(0) == first
    np.testing.assert_array_equal(collector.data.qpos, state)


@pytest.mark.parametrize("failure", ["edge", "airborne", "tipped"])
def test_post_settle_check_rejects_invalid_actual_state(dual_collector, failure):
    collector = dual_collector
    collector.randomize(0)
    joint = mujoco.mj_name2id(collector.model, mujoco.mjtObj.mjOBJ_JOINT, "red_teapot_free")
    address = collector.model.jnt_qposadr[joint]
    if failure == "edge":
        collector.data.qpos[address + 1] = -0.23
    elif failure == "airborne":
        collector.data.qpos[address + 2] += 0.2
    else:
        collector.data.qpos[address + 3 : address + 7] = [np.sqrt(0.5), np.sqrt(0.5), 0, 0]
    mujoco.mj_forward(collector.model, collector.data)
    report = collector.placement.inspect(collector.data)
    assert not report["valid"]
    reason = {"edge": "edge clearance", "airborne": "no tray support", "tipped": "not upright"}
    assert any(reason[failure] in issue for issue in report["issues"])


def test_invalid_settled_layout_is_resampled(dual_collector, monkeypatch):
    collector = dual_collector
    sample = collector._sample_placements
    calls = 0

    def first_at_edge(rng):
        nonlocal calls
        calls += 1
        poses = sample(rng)
        if calls == 1:
            x, _, yaw = poses[2]
            poses[2] = (x, -0.24, yaw)
        return poses

    monkeypatch.setattr(collector, "_sample_placements", first_at_edge)
    result = collector.randomize(0)
    assert result["scene_attempts"] >= 2
    assert result["settled"]["valid"]


def test_impossible_layout_fails_without_committing_an_episode(dual_collector):
    collector = dual_collector
    collector.config["randomization"].update(max_scene_attempts=2, max_placement_attempts=3)
    collector.object_specs[0]["randomization"]["x"] = [2.0, 2.0]
    with pytest.raises(PlacementError, match="after 2 scene attempts.*red_teapot"):
        collector.collect_episode(0)
    assert not list(collector.episodes_root.iterdir())
    assert not (collector.dataset_root / "dataset.jsonl").exists()
    assert collector.status()["current_seed"] is None


def test_primitive_presets_are_also_bounded(tmp_path):
    config = load_config(ROOT / "configs/dual_xhand.yaml")
    for spec in config["objects"]:
        del spec["asset_id"]
    xml = write_scene_xml(config, tmp_path / "primitives.xml")
    model = mujoco.MjModel.from_xml_path(str(xml))
    placement = TrayPlacement(model, config)
    for seed in range(5):
        poses = placement.sample(np.random.default_rng(seed))
        assert len(poses) == 4
    # The asymmetric teapot spout is included, and negative local bottoms affect spawn height.
    assert placement.corners[0][:, 0].min() < -0.11
    assert placement.spawn_height(0) > placement.support_z + 0.06


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("edge_margin_m", -0.02),
        ("edge_margin_m", float("nan")),
        ("minimum_object_gap_m", 0),
        ("max_scene_attempts", 0),
        ("max_placement_attempts", 2.5),
    ],
)
def test_invalid_placement_settings_rejected(key, value):
    config = load_config(ROOT / "configs/dual_xhand.yaml")
    config["randomization"][key] = value
    with pytest.raises(ConfigError, match=key):
        validate_config(config)


def test_invalid_yaw_range_rejected():
    config = load_config(ROOT / "configs/dual_xhand.yaml")
    config["objects"][0]["randomization"]["yaw_deg"] = [30, -30]
    with pytest.raises(ConfigError, match="finite and ascending"):
        validate_config(config)


@pytest.mark.parametrize("timeout", [float("nan"), 0.1, -1.0])
def test_invalid_settle_timeout_rejected(timeout):
    config = load_config(ROOT / "configs/dual_xhand.yaml")
    config["simulation"]["settle_timeout_s"] = timeout
    with pytest.raises(ConfigError, match="settle_timeout_s"):
        validate_config(config)
