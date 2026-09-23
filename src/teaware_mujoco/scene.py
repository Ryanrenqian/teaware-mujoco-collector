from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from .robots import XHAND_JOINT_SUFFIXES
from .teaware_assets import effective_mass_kg, get_teaware_asset, teaware_asset_dir


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


def _add_mesh_teaware_body(
    asset: ET.Element,
    body: ET.Element,
    spec: dict[str, Any],
) -> None:
    name = spec["name"]
    rgba = spec["rgba"]
    metadata = get_teaware_asset(spec["asset_id"])
    asset_root = teaware_asset_dir()
    extents = [float(value) for value in metadata["extents_m"]]
    mass = effective_mass_kg(metadata)
    x, y, z = extents
    inertia = [
        mass * (y * y + z * z) / 12.0,
        mass * (x * x + z * z) / 12.0,
        mass * (x * x + y * y) / 12.0,
    ]
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": _numbers([0.0, 0.0, z / 2.0]),
            "mass": f"{mass:.9g}",
            "diaginertia": _numbers(inertia),
        },
    )

    visual_mesh_name = f"{name}_visual_mesh"
    ET.SubElement(
        asset,
        "mesh",
        {"name": visual_mesh_name, "file": str((asset_root / metadata["visual"]).resolve())},
    )
    _geom(
        body,
        name=f"{name}_visual",
        type="mesh",
        mesh=visual_mesh_name,
        rgba=rgba,
        group=2,
        contype=0,
        conaffinity=0,
        mass=0,
    )

    for index, relative_path in enumerate(metadata["collisions"]):
        mesh_name = f"{name}_collision_mesh_{index:02d}"
        ET.SubElement(
            asset,
            "mesh",
            {"name": mesh_name, "file": str((asset_root / relative_path).resolve())},
        )
        _geom(
            body,
            name=f"{name}_collision_{index:02d}",
            type="mesh",
            mesh=mesh_name,
            rgba=[0, 0, 0, 0],
            group=3,
            contype=1,
            conaffinity=1,
            density=0,
            friction=[0.8, 0.02, 0.001],
        )


def _add_teaware_body(
    asset: ET.Element,
    worldbody: ET.Element,
    spec: dict[str, Any],
    table_top: float,
) -> None:
    name = spec["name"]
    preset = spec["preset"]
    rgba = spec["rgba"]
    body = ET.SubElement(
        worldbody, "body", {"name": name, "pos": _numbers([0.4, 0.0, table_top + 0.12])}
    )
    ET.SubElement(body, "freejoint", {"name": f"{name}_free"})

    if spec.get("asset_id"):
        _add_mesh_teaware_body(asset, body, spec)
        return

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


def xhand_asset_dir() -> Path:
    return Path(str(files("teaware_mujoco").joinpath("assets", "xhand")))


_REFERENCE_ATTRIBUTES = {
    "body1",
    "body2",
    "childclass",
    "class",
    "joint",
    "joint1",
    "joint2",
    "material",
    "mesh",
    "site",
    "tendon",
}


def _strip_template_gripper(template: ET.Element) -> None:
    worldbody = template.find("worldbody")
    if worldbody is not None:
        for parent in worldbody.iter("body"):
            for child in list(parent):
                if child.tag == "body" and child.get("name") == "xarm_gripper_base_link":
                    parent.remove(child)
    for section_name in ("contact", "tendon", "equality"):
        section = template.find(section_name)
        if section is not None:
            section.clear()
    actuator = template.find("actuator")
    if actuator is not None:
        for child in list(actuator):
            if child.get("name") == "gripper":
                actuator.remove(child)


def _namespace_template(template: ET.Element, prefix: str) -> None:
    for mesh in template.findall("./asset/mesh"):
        if "name" not in mesh.attrib:
            mesh.set("name", Path(mesh.attrib["file"]).stem)
        mesh.set("file", str((xarm_asset_dir() / mesh.attrib["file"]).resolve()))
    for element in template.iter():
        if "name" in element.attrib:
            element.set("name", prefix + element.attrib["name"])
        for attribute in _REFERENCE_ATTRIBUTES:
            if attribute in element.attrib:
                element.set(attribute, prefix + element.attrib[attribute])


def _xyz(element: ET.Element | None, attribute: str, default: str = "0 0 0") -> list[float]:
    if element is None:
        return [float(value) for value in default.split()]
    return [float(value) for value in element.get(attribute, default).split()]


def _rpy_quat(rpy: list[float]) -> list[float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _origin_attributes(origin: ET.Element | None) -> dict[str, str]:
    return {
        "pos": _numbers(_xyz(origin, "xyz")),
        "quat": _numbers(_rpy_quat(_xyz(origin, "rpy"))),
    }


def _add_xhand_mesh_asset(
    asset: ET.Element,
    mesh_names: dict[Path, str],
    robot_id: str,
    filename: str,
) -> str:
    mesh_path = (xhand_asset_dir() / filename).resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"xHand mesh does not exist: {mesh_path}")
    if mesh_path not in mesh_names:
        mesh_name = f"{robot_id}_xhand_mesh_{len(mesh_names):02d}_{mesh_path.stem}"
        ET.SubElement(asset, "mesh", {"name": mesh_name, "file": str(mesh_path)})
        mesh_names[mesh_path] = mesh_name
    return mesh_names[mesh_path]


def _add_xhand_geometry(
    body: ET.Element,
    asset: ET.Element,
    mesh_names: dict[Path, str],
    robot_id: str,
    link_name: str,
    geometry_kind: str,
    geometry_index: int,
    source: ET.Element,
) -> None:
    geometry = source.find("geometry")
    if geometry is None:
        return
    suffix = geometry_kind if geometry_index == 0 else f"{geometry_kind}_{geometry_index}"
    attributes = {
        "name": f"{robot_id}_{link_name}_{suffix}",
        **_origin_attributes(source.find("origin")),
        "group": "2" if geometry_kind == "visual" else "3",
    }
    mesh = geometry.find("mesh")
    box = geometry.find("box")
    if mesh is not None:
        attributes.update(
            type="mesh",
            mesh=_add_xhand_mesh_asset(asset, mesh_names, robot_id, mesh.attrib["filename"]),
        )
    elif box is not None:
        attributes.update(
            type="box",
            size=_numbers(np.asarray(_xyz(box, "size"), dtype=np.float64) / 2.0),
        )
    else:
        raise ValueError(f"unsupported xHand geometry on link {link_name}")

    if geometry_kind == "visual":
        color = source.find("./material/color")
        attributes.update(contype="0", conaffinity="0", mass="0")
        if color is not None:
            attributes["rgba"] = color.get("rgba", "0.8 0.8 0.8 1")
    else:
        # Dedicated contact type prevents finger links from blocking each other,
        # while conaffinity keeps contact with ordinary scene/object geoms.
        attributes.update(
            rgba="0 0 0 0",
            friction="0.9 0.02 0.001",
            contype="2",
            conaffinity="1",
        )
    ET.SubElement(body, "geom", attributes)


def _populate_xhand_link(
    *,
    body: ET.Element,
    link: ET.Element,
    children: dict[str, list[ET.Element]],
    links: dict[str, ET.Element],
    asset: ET.Element,
    mesh_names: dict[Path, str],
    robot_id: str,
) -> None:
    link_name = link.attrib["name"]
    inertial = link.find("inertial")
    if inertial is not None:
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        if mass is not None and inertia is not None:
            inertial_origin = inertial.find("origin")
            inertial_rpy = _xyz(inertial_origin, "rpy")
            if not np.allclose(inertial_rpy, 0.0):
                raise ValueError(f"xHand link {link_name} has an unsupported rotated inertia frame")
            ET.SubElement(
                body,
                "inertial",
                {
                    "pos": _numbers(_xyz(inertial_origin, "xyz")),
                    "mass": mass.attrib["value"],
                    "fullinertia": " ".join(
                        inertia.attrib[key] for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
                    ),
                },
            )
    for index, visual in enumerate(link.findall("visual")):
        _add_xhand_geometry(body, asset, mesh_names, robot_id, link_name, "visual", index, visual)
    for index, collision in enumerate(link.findall("collision")):
        _add_xhand_geometry(
            body, asset, mesh_names, robot_id, link_name, "collision", index, collision
        )

    for urdf_joint in children.get(link_name, []):
        child_name = urdf_joint.find("child").attrib["link"]
        child_body = ET.SubElement(
            body,
            "body",
            {
                "name": f"{robot_id}_{child_name}",
                **_origin_attributes(urdf_joint.find("origin")),
            },
        )
        if urdf_joint.get("type") != "fixed":
            limit = urdf_joint.find("limit")
            joint_attributes = {
                "name": f"{robot_id}_{urdf_joint.attrib['name']}",
                "type": "hinge",
                "axis": _numbers(_xyz(urdf_joint.find("axis"), "xyz", "1 0 0")),
                "range": f"{limit.attrib['lower']} {limit.attrib['upper']}",
                # The hardware URDF uses Coulomb friction values comparable to its
                # effort limits. In MuJoCo that stalls position actuators entirely.
                "armature": "0.01",
                "damping": "0.1",
                "frictionloss": "0.01",
            }
            ET.SubElement(child_body, "joint", joint_attributes)
        _populate_xhand_link(
            body=child_body,
            link=links[child_name],
            children=children,
            links=links,
            asset=asset,
            mesh_names=mesh_names,
            robot_id=robot_id,
        )


def _add_xhand(
    link7: ET.Element,
    asset: ET.Element,
    actuator: ET.Element,
    robot_id: str,
    handedness: str,
    force_scale: float,
    actuator_kp: float,
) -> None:
    urdf = ET.parse(xhand_asset_dir() / f"xhand_{handedness}_extended.urdf").getroot()
    links = {link.attrib["name"]: link for link in urdf.findall("link")}
    root_link_name = f"{handedness}_hand_link"
    children: dict[str, list[ET.Element]] = {}
    hand_joints: dict[str, ET.Element] = {}
    for joint in urdf.findall("joint"):
        parent_name = joint.find("parent").attrib["link"]
        child_name = joint.find("child").attrib["link"]
        if parent_name in links and child_name in links:
            children.setdefault(parent_name, []).append(joint)
        if joint.attrib["name"].startswith(f"{handedness}_hand_"):
            hand_joints[joint.attrib["name"]] = joint

    mount = ET.SubElement(
        link7,
        "body",
        {
            "name": f"{robot_id}_xhand_mount",
            "pos": "0.007 0.05 0.005",
            "quat": _numbers(_rpy_quat([0.0, 0.0, -1.5707963])),
        },
    )
    mesh_names: dict[Path, str] = {}
    flange_mesh = _add_xhand_mesh_asset(asset, mesh_names, robot_id, "meshes/xhand_flange.STL")
    ET.SubElement(
        mount,
        "inertial",
        {
            "pos": "0.0501712 0.0581654 0.0000145",
            "mass": "0.4428072",
            "fullinertia": "0.0002726 0.0002431 0.0002042 0 0 0.0000201",
        },
    )
    _geom(
        mount,
        name=f"{robot_id}_xhand_flange_visual",
        type="mesh",
        mesh=flange_mesh,
        rgba=[0.82352941, 0.87058824, 0.98039216, 1.0],
        contype=0,
        conaffinity=0,
        mass=0,
        group=2,
    )
    _geom(
        mount,
        name=f"{robot_id}_xhand_flange_collision",
        type="mesh",
        mesh=flange_mesh,
        rgba=[0.0, 0.0, 0.0, 0.0],
        friction=[0.9, 0.02, 0.001],
        contype=2,
        conaffinity=1,
        group=3,
    )
    hand_rpy = [-1.5707963, 0.0, 3.1415926] if handedness == "left" else [1.5707963, 0.0, 0.0]
    palm = ET.SubElement(
        mount,
        "body",
        {
            "name": f"{robot_id}_{root_link_name}",
            "pos": "0.05 0.045 0.1",
            "quat": _numbers(_rpy_quat(hand_rpy)),
        },
    )
    _populate_xhand_link(
        body=palm,
        link=links[root_link_name],
        children=children,
        links=links,
        asset=asset,
        mesh_names=mesh_names,
        robot_id=robot_id,
    )

    ET.SubElement(
        palm,
        "site",
        {"name": f"{robot_id}_hand_tcp", "pos": "0 0 -0.065", "size": "0.004"},
    )
    for index, suffix in enumerate(XHAND_JOINT_SUFFIXES, start=1):
        urdf_joint = hand_joints[f"{handedness}_hand_{suffix}"]
        limit = urdf_joint.find("limit")
        lower, upper = limit.attrib["lower"], limit.attrib["upper"]
        effort = float(limit.attrib["effort"])
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{robot_id}_xhand_act{index:02d}",
                "joint": f"{robot_id}_{urdf_joint.attrib['name']}",
                "kp": f"{actuator_kp:.9g}",
                "ctrlrange": f"{lower} {upper}",
                "forcerange": _numbers([-effort * force_scale, effort * force_scale]),
            },
        )


def _append_robot(
    *,
    spec: dict[str, Any],
    asset: ET.Element,
    defaults: ET.Element,
    worldbody: ET.Element,
    contact: ET.Element,
    tendon: ET.Element,
    equality: ET.Element,
    actuator: ET.Element,
) -> None:
    template = ET.parse(xarm_asset_dir() / "xarm7.xml").getroot()
    if spec["hand"] == "xhand":
        _strip_template_gripper(template)
    prefix = f"{spec['id']}_"
    _namespace_template(template, prefix)

    template_asset = template.find("asset")
    template_defaults = template.find("default")
    template_worldbody = template.find("worldbody")
    assert template_asset is not None
    assert template_defaults is not None
    assert template_worldbody is not None
    for child in template_asset:
        asset.append(copy.deepcopy(child))
    for child in template_defaults:
        defaults.append(copy.deepcopy(child))

    base = copy.deepcopy(next(iter(template_worldbody)))
    template_position = np.fromstring(base.get("pos", "0 0 0"), sep=" ")
    position = template_position + np.asarray(spec["base_position"], dtype=np.float64)
    base.set("pos", _numbers(position))
    half_yaw = math.radians(float(spec["base_yaw_deg"])) / 2.0
    base.set("quat", _numbers([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]))
    if spec["hand"] == "xhand":
        link7 = next(body for body in base.iter("body") if body.get("name") == f"{prefix}link7")
        _add_xhand(
            link7,
            asset,
            actuator,
            spec["id"],
            spec["handedness"],
            spec["hand_actuator_force_scale"],
            spec["hand_actuator_kp"],
        )
    worldbody.append(base)

    for section_name, destination in (
        ("contact", contact),
        ("tendon", tendon),
        ("equality", equality),
        ("actuator", actuator),
    ):
        source = template.find(section_name)
        if source is None:
            continue
        for child in source:
            destination.append(copy.deepcopy(child))


def build_scene_tree(config: dict[str, Any]) -> ET.ElementTree:
    root = ET.Element("mujoco", {"model": "teaware_collection_scene"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
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

    defaults = ET.SubElement(root, "default")

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

    contact = ET.SubElement(root, "contact")
    tendon = ET.SubElement(root, "tendon")
    equality = ET.SubElement(root, "equality")
    actuator = ET.SubElement(root, "actuator")
    for robot_spec in config["robots"]:
        _append_robot(
            spec=robot_spec,
            asset=asset,
            defaults=defaults,
            worldbody=worldbody,
            contact=contact,
            tendon=tendon,
            equality=equality,
            actuator=actuator,
        )

    for object_spec in config["objects"]:
        _add_teaware_body(asset, worldbody, object_spec, table_top)
    if config["policy"]["type"] == "tro_grasp":
        policy = config["policy"]
        constraint = policy["tro"]["grasp_constraint"]
        if constraint["enabled"]:
            robot = next(spec for spec in config["robots"] if spec["id"] == policy["robot_id"])
            ET.SubElement(
                equality,
                "weld",
                {
                    "name": f"{robot['id']}_{policy['target_object']}_grasp_weld",
                    "body1": f"{robot['id']}_{robot['handedness']}_hand_link",
                    "body2": policy["target_object"],
                    "active": "false",
                    "solref": "0.01 1",
                },
            )
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


def object_spawn_height(spec: dict[str, Any], top_z: float) -> float:
    if spec.get("asset_id"):
        return top_z + 0.025
    half_heights = {"teapot": 0.05, "teacup": 0.035, "pitcher": 0.06, "canister": 0.065}
    return top_z + half_heights[spec["preset"]] + 0.025


def yaw_quaternion(yaw_deg: float) -> np.ndarray:
    half = math.radians(float(yaw_deg)) / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)
