from __future__ import annotations

from pathlib import Path

import mujoco

from teaware_mujoco.scene import write_scene_xml


def test_generated_scene_compiles(tmp_path: Path, tiny_config: dict) -> None:
    path = write_scene_xml(tiny_config, tmp_path / "scene.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.ncam == 1
    assert model.nu == 8
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_teapot") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "white_teacup") >= 0
