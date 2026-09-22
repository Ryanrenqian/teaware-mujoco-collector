from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np


def _numbers(values: list[float] | np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _camera_xyaxes(position: list[float], target: list[float]) -> str:
    position_arr = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - position_arr
    norm = float(np.linalg.norm(forward))
    if norm < 1e-9:
        raise ValueError("camera position and target cannot be identical")
    forward /= norm
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, up))) > 0.98:
        up = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(forward, up)
    x_axis /= np.linalg.norm(x_axis)
    z_axis = -forward
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return _numbers(np.concatenate([x_axis, y_axis]))


def _geom(parent: ET.Element, **attrs: Any) -> ET.Element:
    serialized = {
        key: _numbers(value) if isinstance(value, (list, tuple, np.ndarray)) else str(value)
        for key, value in attrs.items()
    }
    return ET.SubElement(parent, "geom", serialized)


def _capsule(
    parent: ET.Element,
    start: list[float],
    end: list[float],
    radius: float,
    rgba: list[float],
    name: str,
) -> None:
    _geom(
        parent,
        name=name,
        type="capsule",
        fromto=[*start, *end],
        size=[radius],
        rgba=rgba,
        density=650,
        friction=[0.8, 0.02, 0.001],
    )


def _add_handle(body: ET.Element, radius: float, z: float, rgba: list[float], prefix: str) -> None:
    x = radius * 1.25
    _capsule(body, [x, 0.0, z - 0.018], [x + 0.025, 0.0, z], 0.006, rgba, f"{prefix}_handle_low")
    _capsule(
        body, [x + 0.025, 0.0, z], [x + 0.025, 0.0, z + 0.04], 0.006, rgba, f"{prefix}_handle_mid"
    )
    _capsule(
        body, [x + 0.025, 0.0, z + 0.04], [x, 0.0, z + 0.055], 0.006, rgba, f"{prefix}_handle_high"
    )


def _add_teaware_body(worldbody: ET.Element, spec: dict[str, Any], table_top: float) -> None:
    name = spec["name"]
    preset = spec["preset"]
    rgba = spec["rgba"]
    body = ET.SubElement(
        worldbody, "body", {"name": name, "pos": _numbers([0.4, 0.0, table_top + 0.12])}
    )
    ET.SubElement(body, "freejoint", {"name": f"{name}_free"})

    if preset == "teapot":
        _geom(
            body,
            name=f"{name}_body",
            type="cylinder",
            pos=[0, 0, 0],
            size=[0.057, 0.045],
            rgba=rgba,
            density=700,
            friction=[0.8, 0.02, 0.001],
        )
        _geom(
            body,
            name=f"{name}_shoulder",
            type="sphere",
            pos=[0, 0, 0.035],
            size=[0.05],
            rgba=rgba,
            density=250,
        )
        _geom(
            body,
            name=f"{name}_lid",
            type="cylinder",
            pos=[0, 0, 0.075],
            size=[0.032, 0.009],
            rgba=rgba,
            density=500,
        )
        _geom(
            body,
            name=f"{name}_knob",
            type="sphere",
            pos=[0, 0, 0.09],
            size=[0.012],
            rgba=rgba,
            density=500,
        )
        _capsule(body, [-0.04, 0, 0.035], [-0.105, 0, 0.075], 0.014, rgba, f"{name}_spout")
        _add_handle(body, 0.057, 0.015, rgba, name)
    elif preset == "teacup":
        _geom(
            body,
            name=f"{name}_body",
            type="cylinder",
            pos=[0, 0, 0],
            size=[0.036, 0.032],
            rgba=rgba,
            density=650,
            friction=[0.8, 0.02, 0.001],
        )
        _add_handle(body, 0.036, -0.012, rgba, name)
    elif preset == "pitcher":
        _geom(
            body,
            name=f"{name}_body",
            type="cylinder",
            pos=[0, 0, 0],
            size=[0.048, 0.055],
            rgba=rgba,
            density=680,
            friction=[0.8, 0.02, 0.001],
        )
        _capsule(body, [-0.035, 0, 0.04], [-0.07, 0, 0.07], 0.011, rgba, f"{name}_spout")
        _add_handle(body, 0.048, 0.0, rgba, name)
    elif preset == "canister":
        _geom(
            body,
            name=f"{name}_body",
            type="cylinder",
            pos=[0, 0, 0],
            size=[0.042, 0.06],
            rgba=rgba,
            density=700,
            friction=[0.8, 0.02, 0.001],
        )
        _geom(
            body,
            name=f"{name}_lid",
            type="cylinder",
            pos=[0, 0, 0.066],
            size=[0.044, 0.008],
            rgba=rgba,
            density=450,
        )


def xarm_asset_dir() -> Path:
    return Path(str(files("teaware_mujoco").joinpath("assets", "ufactory_xarm7")))


def build_scene_tree(config: dict[str, Any]) -> ET.ElementTree:
    root = ET.Element("mujoco", {"model": "teaware_collection_scene"})
    ET.SubElement(root, "include", {"file": str((xarm_asset_dir() / "xarm7.xml").resolve())})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": f"{config['simulation']['timestep']:.9g}",
            "gravity": "0 0 -9.81",
            "integrator": "implicitfast",
        },
    )
    ET.SubElement(root, "statistic", {"center": "0.4 0 0.35", "extent": "1.1"})

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.55 0.55 0.55", "ambient": "0.35 0.35 0.35", "specular": "0.1 0.1 0.1"},
    )
    ET.SubElement(
        visual,
        "global",
        {
            "azimuth": "135",
            "elevation": "-25",
            "offwidth": str(config["renderer"]["width"]),
            "offheight": str(config["renderer"]["height"]),
        },
    )

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.82 0.86 0.9",
            "rgb2": "0.18 0.22 0.27",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": "floor_tex",
            "builtin": "checker",
            "rgb1": "0.2 0.22 0.24",
            "rgb2": "0.32 0.35 0.38",
            "width": "256",
            "height": "256",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "floor_mat",
            "texture": "floor_tex",
            "texrepeat": "6 6",
            "texuniform": "true",
            "reflectance": "0.05",
        },
    )

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "key",
            "pos": "0.2 -0.3 1.8",
            "dir": "0.1 0.1 -1",
            "directional": "true",
            "diffuse": "0.85 0.85 0.82",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "fill",
            "pos": "0.8 0.6 1.2",
            "dir": "-0.4 -0.2 -1",
            "directional": "true",
            "diffuse": "0.45 0.48 0.55",
        },
    )
    _geom(
        worldbody,
        name="floor",
        type="plane",
        pos=[0, 0, -0.001],
        size=[0, 0, 0.05],
        material="floor_mat",
        contype=1,
        conaffinity=1,
    )

    table = config["table"]
    _geom(
        worldbody,
        name="tea_table",
        type="box",
        pos=table["center"],
        size=[value / 2.0 for value in table["size"]],
        rgba=table["rgba"],
        friction=[0.9, 0.02, 0.001],
        contype=1,
        conaffinity=1,
    )
    table_top = float(table["center"][2]) + float(table["size"][2]) / 2.0
    _geom(
        worldbody,
        name="tea_tray",
        type="box",
        pos=[0.46, 0.0, table_top + 0.008],
        size=[0.27, 0.23, 0.008],
        rgba=[0.16, 0.19, 0.18, 1],
        friction=[0.9, 0.02, 0.001],
    )

    for object_spec in config["objects"]:
        _add_teaware_body(worldbody, object_spec, table_top)
    for camera in config["cameras"]:
        ET.SubElement(
            worldbody,
            "camera",
            {
                "name": camera["name"],
                "pos": _numbers(camera["position"]),
                "xyaxes": _camera_xyaxes(camera["position"], camera["target"]),
                "fovy": f"{camera['fovy']:.9g}",
            },
        )
    return ET.ElementTree(root)


def write_scene_xml(config: dict[str, Any], destination: str | Path) -> Path:
    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    tree = build_scene_tree(config)
    ET.indent(tree, space="  ")
    tree.write(destination_path, encoding="utf-8", xml_declaration=True)
    return destination_path


def table_top_z(config: dict[str, Any]) -> float:
    table = config["table"]
    return float(table["center"][2]) + float(table["size"][2]) / 2.0


def object_spawn_height(preset: str, top_z: float) -> float:
    half_heights = {"teapot": 0.05, "teacup": 0.035, "pitcher": 0.06, "canister": 0.065}
    return top_z + half_heights[preset] + 0.025


def yaw_quaternion(yaw_deg: float) -> np.ndarray:
    half = math.radians(float(yaw_deg)) / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)
