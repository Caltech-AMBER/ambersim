import jax.numpy as jnp
import mujoco as mj
import numpy as np
import pinocchio as pin
import pytest
from mujoco import mjx

from ambersim import ROOT
from ambersim.utils.introspection_utils import get_joint_names
from ambersim.utils.io_utils import load_mj_model_from_file, mj_to_mjx_model_and_data


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

    # setting dummy positions
    _qpos_mj = []
    for mjjn in mj_joint_names:
        if mjjn != "freejoint":
            _qpos_mj.append([0.5])  # arbitrary
        else:
            # floating base state convention: (translation[x, y, z], quat[w, x, y, z])
            _qpos_mj.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    _qpos_pin = []
    for pjn in pin_joint_names:
        if pjn not in ["freejoint", "universe"]:
            _qpos_pin.append([0.5])  # arbitrary
        elif pjn != "universe":
            # floating base state convention: (translation[x, y, z], quat[x, y, z, w])
            _qpos_pin.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    qpos_mj = jnp.array(np.concatenate(_qpos_mj))
    qpos_pin = np.concatenate(_qpos_pin)

    # return shared variables
    return mjx_model, mjx_data, mj_joint_names, pin_model, pin_data, pin_joint_names, qpos_mj, qpos_pin


def test_fk(shared_variables):
    """Tests forward kinematics.

    Checks against pinocchio. The two main bookkeeping things are (1) the joint names and (2) the different quaternion convention.
    """
    # loading global variables
    mjx_model, mjx_data, mj_joint_names, pin_model, pin_data, pin_joint_names, qpos_mj, qpos_pin = shared_variables

    # running FK test
    mjx_data = mjx.kinematics(mjx_model, mjx_data.replace(qpos=qpos_mj))
    pin.forwardKinematics(pin_model, pin_data, qpos_pin)

    for i, mjjn in enumerate(mj_joint_names):
        j = pin_joint_names.index(mjjn)
        assert np.allclose(
            np.array(mjx_data.xanchor[i, :]),
            np.array(pin_data.oMi[j].translation),
            rtol=1e-6,
            atol=1e-6,  # loosen atol a bit
        )


def test_fd():
    """Tests forward dynamics."""


def test_id():
    """Tests inverse dynamics."""
