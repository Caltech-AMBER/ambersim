from typing import List, Tuple

import jax
import mujoco as mj
import numpy as np
import pinocchio as pin
from mujoco import mjx
from mujoco.mjx._src.scan import _q_jointid

from ambersim.utils.introspection_utils import get_joint_names

"""Utils for converting quantities between mujoco/mjx and pinocchio."""


def get_joint_orders(
    mj_model: mj.MjModel,
    pin_model: pin.Model,
) -> Tuple[List[int], List[int]]:
    """Computes the mapping from joint orders in mujoco and pinocchio.

    Pinocchio maintains one extra joint relative to mujoco (the "universe" joint).
    The outputs satisfy the following two tests:

    assert [mj_joint_names[i] if i is not None else "universe" for i in mj2pin] == pin_joint_names
    assert [pin_joint_names[i] for i in pin2mj if i is not None] == mj_joint_names

    Args:
        mj_model: The MuJoCo model.
        pin_model: The Pinocchio model.

    Returns:
        pin2mj: The mapping from pinocchio joint order to mujoco joint order.
        mj2pin: The mapping from mujoco joint order to pinocchio joint order.
    """
    mj_joint_names = get_joint_names(mj_model)
    pin_joint_names = list(pin_model.names)

    mj2pin = [mj_joint_names.index(a) if a in mj_joint_names else None for a in pin_joint_names]
    pin2mj = [pin_joint_names.index(a) if a in pin_joint_names else None for a in mj_joint_names]

    return mj2pin, pin2mj


def mjx_xanchor_to_pin(mj2pin: List[int], mjx_data: mjx.Data) -> np.ndarray:
    """Converts the mujoco joint anchor positions to pinocchio joint placements.

    Args:
        mj2pin: The mapping from mujoco joint order to pinocchio joint order.
        mjx_data: The mujoco data.

    Returns:
        xanchor_pin: The pinocchio joint placements.
    """
    xanchors = np.array(mjx_data.xanchor)
    _xanchor_pin = []
    for mj_idx in mj2pin:
        if mj_idx is not None:
            _xanchor_pin.append(xanchors[mj_idx, :])
        else:
            _xanchor_pin.append(np.zeros(3))
    return np.stack(_xanchor_pin)


def pin_xanchor_to_mjx(pin2mj: List[int], pin_data: pin.Data) -> np.ndarray:
    """Converts the pinocchio joint placements to mujoco joint anchor positions.

    Args:
        pin2mj: The mapping from pinocchio joint order to mujoco joint order.
        pin_data: The pinocchio data.

    Returns:
        xanchor_mj: The mujoco joint anchor positions.
    """
    _xanchor_mj = []
    for pin_idx in pin2mj:
        if pin_idx is not None:
            _xanchor_mj.append(pin_data.oMi[pin_idx].translation)
        else:
            _xanchor_mj.append(np.zeros(3))
    return np.stack(_xanchor_mj)


def mjx_qpos_to_pin(qpos_mjx: jax.Array, mjx_model: mjx.Model, mj2pin: List[int]) -> np.ndarray:
    """Converts the mujoco joint positions to pinocchio joint positions.

    Args:
        qpos_mjx: The mjx joint positions.
        mjx_model: The mjx model.
        mj2pin: The mapping from mujoco joint order to pinocchio joint order.

    Returns:
        qpos_pin: The pinocchio joint positions.
    """
    qpos_mjx = np.array(qpos_mjx)
    _qpos_pin = []
    jnt_types = mjx_model.jnt_type
    for mj_idx in mj2pin:
        # ignore Nones
        if mj_idx is None:
            continue

        # check how to append based on joint type
        jnt_type = jnt_types[mj_idx]
        startidx = mjx_model.jnt_qposadr[mj_idx]

        if jnt_type == 0:  # FREE, 7-dimensional
            _qpos_pin.append(qpos_mjx[startidx : startidx + 3])
            quat_wxyz = qpos_mjx[startidx + 3 : startidx + 7]
            quat_xyzw = np.concatenate((quat_wxyz[1:], quat_wxyz[:1]))
            _qpos_pin.append(quat_xyzw)
        elif jnt_type == 1:  # BALL, 4-dimensional
            quat_wxyz = qpos_mjx[startidx : startidx + 4]
            quat_xyzw = np.concatenate((quat_wxyz[1:], quat_wxyz[:1]))
            _qpos_pin.append(quat_xyzw)
        elif jnt_type == 2:  # SLIDE, 3-dimensional
            _qpos_pin.append(qpos_mjx[startidx : startidx + 3])
        else:  # HINGE, 1-dimensional
            _qpos_pin.append(qpos_mjx[startidx : startidx + 1])

    return np.concatenate(_qpos_pin)


def mjx_qvel_to_pin(qvel_mjx: jax.Array, mjx_model: mjx.Model, mj2pin: List[int]) -> np.ndarray:
    """Converts the mujoco joint velocities to pinocchio joint velocities.

    Notice that there is one less coordinate for rotational velocities compared to quaternion coordinates.

    Args:
            qvel_mjx: The mjx joint velocities.
            mjx_model: The mjx model.
            mj2pin: The mapping from mujoco joint order to pinocchio joint order.

    Returns:
            qvel_pin: The pinocchio joint velocities.
    """
    qvel_mjx = np.array(qvel_mjx)
    _qvel_pin = []
    jnt_types = mjx_model.jnt_type
    for mj_idx in mj2pin:
        # ignore Nones
        if mj_idx is None:
            continue

        # check how to append based on joint type
        jnt_type = jnt_types[mj_idx]
        startidx = mjx_model.jnt_dofadr[mj_idx]

        if jnt_type == 0:  # FREE, 6-dimensional
            _qvel_pin.append(qvel_mjx[startidx : startidx + 3])
            _qvel_pin.append(qvel_mjx[startidx + 3 : startidx + 6])
        elif jnt_type == 1:  # BALL, 3-dimensional
            _qvel_pin.append(qvel_mjx[startidx : startidx + 3])
        elif jnt_type == 2:  # SLIDE, 1-dimensional
            _qvel_pin.append(qvel_mjx[startidx : startidx + 1])
        else:  # HINGE, 1-dimensional
            _qvel_pin.append(qvel_mjx[startidx : startidx + 1])

    return np.concatenate(_qvel_pin)
