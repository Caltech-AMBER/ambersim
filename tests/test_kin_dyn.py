import mujoco as mj
import numpy as np
import pinocchio as pin
from mujoco import mjx

from ambersim import ROOT
from ambersim.utils.io_utils import load_mjx_model_and_data_from_file


def test_fk():
    """Tests forward kinematics."""
    urdf_path = ROOT + "/models/barrett_hand/bh280.urdf"

    # from ambersim.utils.io_utils import load_mj_model_from_file

    # mj_model = load_mj_model_from_file(urdf_path, force_float=True)

    # mjx
    # floating base state convention: (translation[x, y, z], quat[w, x, y, z])
    mjx_model, mjx_data = load_mjx_model_and_data_from_file(urdf_path, force_float=True)
    qpos_new = mjx_data.qpos.at[7:].set(0.5)  # set all non-quat states to 0.5
    data_new = mjx_data.replace(qpos=qpos_new)
    data_post_kin = mjx.kinematics(mjx_model, data_new)
    xpos_mjx_post_kin = data_post_kin.xpos
    print(xpos_mjx_post_kin)

    # pinocchio
    # floating base state convention: (translation[x, y, z], quat[x, y, z, w])
    pin_model = pin.buildModelFromUrdf(urdf_path)
    pin_model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "freejoint")
    pin_data = pin_model.createData()
    breakpoint()
    qpos_quat_xyzw = np.roll(np.array(qpos_new)[3:7], -1, axis=0)
    pin_qpos = np.concatenate((np.array(qpos_new)[7:], np.array(qpos_new)[:7]))
    pin_qpos[-4:] = qpos_quat_xyzw
    pin.forwardKinematics(pin_model, pin_data, pin_qpos)
    pin.updateFramePlacements(pin_model, pin_data)
    for oMi in pin_data.oMi:
        print(oMi.translation)
    breakpoint()

    # # mjx
    # # floating base state convention: (translation[x, y, z], quat[w, x, y, z])
    # mjx_model, mjx_data = load_mjx_model_and_data_from_file(urdf_path, force_float=False)
    # qpos_new = mjx_data.qpos.at[:].set(0.5)
    # data_new = mjx_data.replace(qpos=qpos_new)
    # data_post_kin = mjx.kinematics(mjx_model, data_new)
    # xpos_mjx_post_kin = data_post_kin.xpos

    # # pinocchio
    # # floating base state convention: (translation[x, y, z], quat[x, y, z, w])
    # pin_model = pin.buildModelFromUrdf(urdf_path)
    # pin_data = pin_model.createData()
    # pin_qpos = np.array(qpos_new)
    # pin.forwardKinematics(pin_model, pin_data, pin_qpos)
    # pin.updateFramePlacements(pin_model, pin_data)
    # breakpoint()


def test_fd():
    """Tests forward dynamics."""


def test_id():
    """Tests inverse dynamics."""
