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


def test_generated_scene_uses_visual_and_convex_collision_teaware_meshes(
    tmp_path: Path, tiny_config: dict
) -> None:
    path = write_scene_xml(tiny_config, tmp_path / "scene.xml")
    root = ET.parse(path).getroot()
    meshes = {mesh.attrib["name"]: Path(mesh.attrib["file"]) for mesh in root.findall("./asset/mesh")}

    for spec in tiny_config["objects"]:
        name = spec["name"]
        body = root.find(f"./worldbody/body[@name='{name}']")
        assert body is not None
        assert body.find("inertial") is not None
        assert body.find(f"geom[@name='{name}_visual']") is not None
        assert len(body.findall("geom[@type='mesh'][@group='3']")) == 16
        assert meshes[f"{name}_visual_mesh"].name == "visual.obj"
        assert all(path.is_file() for mesh_name, path in meshes.items() if mesh_name.startswith(name))
        assert body.find(f"geom[@name='{name}_body']") is None


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
    assert len(xhand_meshes) == 50
    assert all(path.is_file() for path in xhand_meshes)
    assert any(path.name == "right_hand_link.STL" for path in xhand_meshes)
    assert any(path.name == "left_hand_link.STL" for path in xhand_meshes)
    assert sum(path.name == "xhand_flange.STL" for path in xhand_meshes) == 2

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
        inertial = mount.find("inertial")
        assert inertial.attrib["mass"] == "0.4428072"
        assert mount.find(f"geom[@name='{robot_id}_xhand_flange_visual']") is not None
        assert mount.find(f"geom[@name='{robot_id}_xhand_flange_collision']") is not None
        assert mount.find(f"body[@name='{robot_id}_{handedness}_hand_link']") is not None

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

        finger_joint = bend.find("joint")
        assert finger_joint.attrib["armature"] == "0.01"
        assert finger_joint.attrib["damping"] == "0.1"
        assert finger_joint.attrib["frictionloss"] == "0.01"


def test_xhand_position_actuators_track_closed_targets(tmp_path: Path) -> None:
    from teaware_mujoco.config import load_config
    from teaware_mujoco.robots import XHAND_OPEN_Q

    config = load_config(REPO_ROOT / "configs/single_xhand.yaml")
    path = write_scene_xml(config, tmp_path / "single.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    hand_joint_ids = np.asarray(
        [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"right_arm_right_hand_{suffix}",
            )
            for suffix in (
                "thumb_bend_joint",
                "thumb_rota_joint1",
                "thumb_rota_joint2",
                "index_bend_joint",
                "index_joint1",
                "index_joint2",
                "mid_joint1",
                "mid_joint2",
                "ring_joint1",
                "ring_joint2",
                "pinky_joint1",
                "pinky_joint2",
            )
        ],
        dtype=np.int32,
    )
    actuator_ids = np.asarray(
        [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"right_arm_xhand_act{index:02d}"
            )
            for index in range(1, 13)
        ],
        dtype=np.int32,
    )
    arm_joint_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"right_arm_joint{index}")
            for index in range(1, 8)
        ],
        dtype=np.int32,
    )
    arm_actuator_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"right_arm_act{index}")
            for index in range(1, 8)
        ],
        dtype=np.int32,
    )
    home_q = np.asarray(config["robots"][0]["home_q"])
    data.qpos[model.jnt_qposadr[arm_joint_ids]] = home_q
    data.ctrl[arm_actuator_ids] = home_q
    qpos_addresses = model.jnt_qposadr[hand_joint_ids]
    data.qpos[qpos_addresses] = XHAND_OPEN_Q
    target = np.asarray([1.65, 0.6, 0.35, 0.0, 0.7, 0.5, 0.7, 0.5, 0.7, 0.5, 0.7, 0.5])
    data.ctrl[actuator_ids] = target
    mujoco.mj_forward(model, data)

    for _ in range(round(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    np.testing.assert_allclose(data.qpos[qpos_addresses], target, atol=0.03)


def test_tro_xhand_actuators_use_configured_grasp_strength(tmp_path: Path) -> None:
    from teaware_mujoco.config import load_config

    config = load_config(REPO_ROOT / "configs/tro_xhand_mock.yaml")
    path = write_scene_xml(config, tmp_path / "tro.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_arm_xhand_act06"
    )
    assert model.actuator_gainprm[actuator_id, 0] == 20.0
    np.testing.assert_allclose(model.actuator_forcerange[actuator_id], [-1.0, 1.0])
