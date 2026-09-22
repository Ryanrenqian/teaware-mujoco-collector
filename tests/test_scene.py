from __future__ import annotations

from pathlib import Path

import mujoco

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
