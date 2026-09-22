from __future__ import annotations

from typing import Final

ARM_JOINT_SUFFIXES: Final = tuple(f"joint{index}" for index in range(1, 8))
ARM_ACTUATOR_SUFFIXES: Final = tuple(f"act{index}" for index in range(1, 8))

# Matches the q12 ordering used by the xHand planning and hardware stack in waic-demo4.
XHAND_JOINT_SUFFIXES: Final = (
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
XHAND_LOWER: Final = (
    0.0,
    -1.05,
    -0.175,
    -0.175,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
XHAND_UPPER: Final = (
    1.83,
    1.57,
    1.83,
    0.175,
    1.92,
    1.92,
    1.92,
    1.94,
    1.92,
    1.92,
    1.92,
    1.92,
)
XHAND_OPEN_Q: Final = (1.45, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
MAX_HAND_DOF: Final = len(XHAND_JOINT_SUFFIXES)

HAND_TYPES: Final = frozenset({"gripper", "xhand"})
HANDEDNESS: Final = frozenset({"left", "right"})


def hand_dof(hand_type: str) -> int:
    if hand_type == "gripper":
        return 1
    if hand_type == "xhand":
        return MAX_HAND_DOF
    raise ValueError(f"unsupported hand type: {hand_type}")


def xhand_joint_names(robot_id: str, handedness: str) -> list[str]:
    return [f"{robot_id}_{handedness}_hand_{suffix}" for suffix in XHAND_JOINT_SUFFIXES]


def xhand_actuator_names(robot_id: str) -> list[str]:
    return [f"{robot_id}_xhand_act{index:02d}" for index in range(1, MAX_HAND_DOF + 1)]
