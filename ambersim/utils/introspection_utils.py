from typing import List

import mujoco as mj

"""Useful utils for introspecting a mujoco model."""


def get_actuator_names(model: mj.MjModel) -> List[str]:
    """Returns a list of all actuator names in a mujoco (NOT mjx) model."""
    return [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]


def get_equality_names(model: mj.MjModel) -> List[str]:
    """Returns a list of all equality constraint names in a mujoco (NOT mjx) model."""
    return [mj.mj_id2name(model, mj.mjtObj.mjOBJ_EQUALITY, i) for i in range(model.neq)]


def get_geom_names(model: mj.MjModel) -> List[str]:
    """Returns a list of all geom names in a mujoco (NOT mjx) model."""
    return [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]


def get_joint_names(model: mj.MjModel) -> List[str]:
    """Returns a list of all joint names in a mujoco (NOT mjx) model."""
    return [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
