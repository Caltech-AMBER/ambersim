from typing import List

import mujoco as mj


def get_geom_names(model: mj.MjModel) -> List[str]:
    """Returns a list of all geom names in a mujoco (NOT mjx) model."""
    return [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]


def get_joint_names(model: mj.MjModel) -> List[str]:
    """Returns a list of all joint names in a mujoco (NOT mjx) model."""
    return [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
