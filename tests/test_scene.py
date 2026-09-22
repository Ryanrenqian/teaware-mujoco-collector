from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from teaware_mujoco.scene import write_scene_xml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_generated_scene_compiles(tmp_path: Path, tiny_config: dict) -> None:
    path = write_scene_xml(tiny_config, tmp_path / "scene.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.ncam == 1
    assert model.nu == 8
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_teapot") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "white_teacup") >= 0


def test_dual_xhand_scene_has_namespaced_arms_and_hands(tmp_path: Path) -> None:
    from teaware_mujoco.config import load_config

    config = load_config(REPO_ROOT / "configs/dual_xhand.yaml")
    path = write_scene_xml(config, tmp_path / "dual.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nu == 38
    for robot_id, handedness in (("left_arm", "left"), ("right_arm", "right")):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot_id}_joint7") >= 0
        assert (
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{robot_id}_{handedness}_hand_thumb_bend_joint",
            )
            >= 0
        )
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{robot_id}_hand_tcp") >= 0


def test_xhand_scene_uses_real_meshes_and_handed_urdf_kinematics(tmp_path: Path) -> None:
    from teaware_mujoco.config import load_config

    config = load_config(REPO_ROOT / "configs/dual_xhand.yaml")
    path = write_scene_xml(config, tmp_path / "dual.xml")
    root = ET.parse(path).getroot()

    mesh_files = [Path(mesh.attrib["file"]) for mesh in root.findall("./asset/mesh")]
    xhand_meshes = [path for path in mesh_files if "assets/xhand/meshes" in path.as_posix()]
    assert len(xhand_meshes) == 48
    assert all(path.is_file() for path in xhand_meshes)
    assert any(path.name == "right_hand_link.STL" for path in xhand_meshes)
    assert any(path.name == "left_hand_link.STL" for path in xhand_meshes)

    bodies = {body.attrib["name"]: body for body in root.iter("body")}
    for robot_id, handedness, bend_y, bend_axis, thumb_quat in (
        (
            "left_arm",
            "left",
            -0.0095,
            -1.0,
            [0.991239539, -0.13049947, 0.00265603086, -0.0201745096],
        ),
        (
            "right_arm",
            "right",
            0.0095,
            1.0,
            [0.991239539, 0.13049947, 0.00265603086, 0.0201745096],
        ),
    ):
        mount = bodies[f"{robot_id}_xhand_mount"]
        np.testing.assert_allclose(
            np.fromstring(mount.attrib["pos"], sep=" "), [0.007, 0.05, 0.005]
        )

        bend = bodies[f"{robot_id}_{handedness}_hand_thumb_bend_link"]
        np.testing.assert_allclose(
            np.fromstring(bend.attrib["pos"], sep=" "), [0.0228, bend_y, -0.0305]
        )
        np.testing.assert_allclose(
            np.fromstring(bend.find("joint").attrib["axis"], sep=" "), [0.0, 0.0, bend_axis]
        )

        thumb = bodies[f"{robot_id}_{handedness}_hand_thumb_rota_link1"]
        np.testing.assert_allclose(
            np.fromstring(thumb.attrib["quat"], sep=" "), thumb_quat, atol=1e-8
        )
