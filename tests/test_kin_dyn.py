import jax.numpy as jnp
import mujoco as mj
import numpy as np
import pinocchio as pin
import pytest
from mujoco import mjx

from ambersim import ROOT
from ambersim.utils.introspection_utils import get_joint_names
from ambersim.utils.io_utils import load_mj_model_from_file, mj_to_mjx_model_and_data
from ambersim.utils.pin_utils import (
    get_joint_orders,
    mjx_qpos_to_pin,
    mjx_qvel_to_pin,
    mjx_xanchor_to_pin,
    pin_xanchor_to_mjx,
)

"""
The tests here check the mjx calculations against pinocchio (and also demonstrate how to use the kin/dyn funcs
independently of environments). The two main bookkeeping things are (1) the joint names and (2) the different
quaternion convention.

TODO(ahl):
* test correctness of batched data
* use more convincing nonzero values for dynamics tests
"""


@pytest.fixture
def shared_variables():
    """Configuration method for global variables."""
    # run all tests on the barrett hand
    urdf_path = ROOT + "/models/barrett_hand/bh280.urdf"

    # loading mjx models
    mj_model = load_mj_model_from_file(urdf_path, force_float=True)
    mjx_model, mjx_data = mj_to_mjx_model_and_data(mj_model)
    mj_joint_names = get_joint_names(mj_model)

    # loading pinocchio models
    pin_model = pin.buildModelFromUrdf(urdf_path)
    pin_model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "freejoint")
    pin_data = pin_model.createData()
    pin_joint_names = list(pin_model.names)

    # relating mujoco and pinocchio joint orders
    mj2pin, pin2mj = get_joint_orders(mj_model, pin_model)

    # setting dummy positions
    _qpos_mj = []
    for mjjn in mj_joint_names:
        if mjjn != "freejoint":
            _qpos_mj.append([np.random.randn()])  # arbitrary
        else:
            # floating base state convention: (translation[x, y, z], quat[w, x, y, z])
            # TODO(ahl): also randomize rotation
            _qpos_mj.append([np.random.randn(), np.random.randn(), np.random.randn(), 1.0, 0.0, 0.0, 0.0])
    qpos_mjx = jnp.array(np.concatenate(_qpos_mj))
    qpos_pin = mjx_qpos_to_pin(qpos_mjx, mjx_model, mj2pin)  # accounts for potentially different joint ordering

    # setting dummy velocities
    # qvel_mjx = jnp.array(np.random.randn(mjx_model.nv))
    # qvel_pin = mjx_qvel_to_pin(qvel_mjx, mjx_model, mj2pin)  # accounts for potentially different joint ordering

    # return shared variables
    return (
        mj_model,
        mjx_model,
        mjx_data,
        mj_joint_names,
        pin_model,
        pin_data,
        pin_joint_names,
        qpos_mjx,
        qpos_pin,
        mj2pin,
        pin2mj,
    )


def test_joint_orders(shared_variables):
    """Tests joint ordering."""
    mj_model = shared_variables[0]
    mj_joint_names = shared_variables[3]
    pin_model = shared_variables[4]
    pin_joint_names = shared_variables[6]
    mj2pin, pin2mj = get_joint_orders(mj_model, pin_model)
    assert [mj_joint_names[i] if i is not None else "universe" for i in mj2pin] == pin_joint_names
    assert [pin_joint_names[i] for i in pin2mj if i is not None] == mj_joint_names


def test_fk(shared_variables):
    """Tests forward kinematics. Also implicitly tests conversion functions between mjx and pin."""
    # loading global variables
    (
        mj_model,
        mjx_model,
        mjx_data,
        mj_joint_names,
        pin_model,
        pin_data,
        pin_joint_names,
        qpos_mjx,
        qpos_pin,
        mj2pin,
        pin2mj,
    ) = shared_variables

    # running FK test
    mjx_data = mjx.kinematics(mjx_model, mjx_data.replace(qpos=qpos_mjx))
    pin.forwardKinematics(pin_model, pin_data, qpos_pin)

    # testing in one direction (mjx coords -> pin)
    xanchors_pin = mjx_xanchor_to_pin(mj2pin, mjx_data)
    for i, x in enumerate(xanchors_pin):
        assert np.allclose(
            np.array(x),
            np.array(pin_data.oMi[i].translation),
            rtol=1e-6,
            atol=1e-6,  # loosen atol a bit
        )

    # testing other direction (pin coords -> mjx)
    xanchors_mjx = mjx_data.xanchor
    xanchors_mjx_converted = pin_xanchor_to_mjx(pin2mj, pin_data)
    for i, xanchor_mjx in enumerate(xanchors_mjx):
        if mj2pin[i] is not None:
            assert np.allclose(
                np.array(xanchor_mjx),
                np.array(xanchors_mjx_converted[i, :]),
                rtol=1e-6,
                atol=1e-6,  # loosen atol a bit
            )


# def test_fd(shared_variables):
#     """Tests forward dynamics."""
#     # mj_model, mjx_model, mjx_data, mj_joint_names, pin_model, pin_data, pin_joint_names, qpos_mjx, qpos_pin, mj2pin, pin2mj = shared_variables

#     # # running FD test
#     #
#     # dv_mjx = np.array(mjx_data.qacc)

#     # M = pin.crba(pin_model, pin_data, qpos_pin)mjx_data = mjx.forward(mjx_model, mjx_data.replace(qpos=qpos_mjx))
#     # qvel_pin = np.zeros(pin_model.nv)  # [TODO] make this more exotic
#     # zero_accel = np.zeros(pin_model.nv)
#     # tau0 = pin.rnea(pin_model, pin_data, qpos_pin, qvel_pin, zero_accel)
#     # # [TODO] add applied forces tau to pinocchio and mjx for the test, check about gravity
