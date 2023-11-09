from pathlib import Path
from typing import Optional, Tuple, Union

import coacd
import mujoco as mj
import numpy as np
import trimesh
from dm_control import mjcf
from mujoco import mjx

from ambersim import ROOT
from ambersim.utils._internal_utils import _check_filepath
from ambersim.utils.conversion_utils import save_model_xml


def _modify_robot_float_base(filepath: Union[str, Path]) -> mj.MjModel:
    """Modifies a robot to have a floating base if it doesn't already."""
    # loading current robot
    assert str(filepath).split(".")[-1] == "xml"
    robot = mjcf.from_file(filepath, model_dir="/".join(filepath.split("/")[:-1]))

    # only add free joint if the first body after worldbody has no freejoints
    assert robot.worldbody is not None
    if robot.worldbody.body[0].freejoint is None:
        # creates a new robot object and attaches the current robot to it freely
        arena = mjcf.RootElement(model=robot.model)
        attachment_frame = arena.attach(robot)
        attachment_frame.add("freejoint", name="freejoint")
        robot = arena
        assert robot.worldbody is not None  # pyright typechecking
        assert robot.worldbody.body[0].inertial is not None  # checking for non-physical parsing errors

    # extracts the mujoco model from the dm_mujoco Physics object
    physics = mjcf.Physics.from_mjcf_model(robot)
    assert physics is not None  # pyright typechecking
    model = physics.model.ptr
    return model


def load_mj_model_from_file(
    filepath: Union[str, Path],
    force_float: bool = False,
    solver: Union[str, mj.mjtSolver] = mj.mjtSolver.mjSOL_CG,
    iterations: Optional[int] = None,
    ls_iterations: Optional[int] = None,
) -> mj.MjModel:
    """Loads a mujoco model from a filepath.

    Args:
        filepath: A path to a URDF or MJCF file. This can be global, local, or with respect to the repository root.
        force_float: Whether to forcibly float the base of the robot.
        solver: The type of solver to use. For now, mjx only supports CG.
        iterations: The number of solver iterations to use. If unspecified, mujoco uses 100.
        ls_iterations: The number of line search iterations to use. If unspecified, mujoco uses 50.

    Returns:
        mj_model: A mujoco model.
    """
    # TODO(ahl): once we allow installing mujoco from source, update this to allow the Newton solver + update default
    if isinstance(solver, str):
        if solver.lower() == "cg":
            solver = mj.mjtSolver.SOL_CG
        else:
            raise ValueError("Solver must be one of: ['cg']!")
    elif isinstance(solver, mj.mjtSolver):
        assert solver in [mj.mjtSolver.mjSOL_CG]

    filepath = _check_filepath(filepath)

    # loading the model and data. check whether freejoint is added forcibly
    if force_float:
        # check that the file is an XML. if not, save as xml temporarily
        if str(filepath).split(".")[-1] != "xml":
            output_name = "/".join(str(filepath).split("/")[:-1]) + "/_temp_xml_model.xml"
            save_model_xml(filepath, output_name=output_name)
            mj_model = _modify_robot_float_base(output_name)
            Path.unlink(output_name)
        else:
            mj_model = _modify_robot_float_base(filepath)
    else:
        mj_model = mj.MjModel.from_xml_path(filepath)

    # setting solver options
    mj_model.opt.solver = solver
    if iterations is not None:
        mj_model.opt.iterations = iterations
    if ls_iterations is not None:
        mj_model.opt.ls_iterations = ls_iterations

    return mj_model


def mj_to_mjx_model_and_data(mj_model: mj.MjModel) -> Tuple[mjx.Model, mjx.Data]:
    """Converts a mujoco model to an mjx (model, data) pair."""
    try:
        mjx_model = mjx.device_put(mj_model)
        mjx_data = mjx.make_data(mjx_model)
        return mjx_model, mjx_data
    except NotImplementedError as e:
        extended_msg = """
        If you are seeing this, there are many potential errors.

        (1) There are some URDF convex primitives that aren't compatible with mjx's convex collision checking.
            See: https://github.com/google-deepmind/mujoco/blob/57e6940f579484adf34eebedc51279a818909f34/mjx/mujoco/mjx/_src/collision_driver.py#L47-L62.

        (2) You are specifying an elliptic instead of pyramidal friction cone.

        (3) You are attempting to port over any number of development or unsupported features, listed here:
            https://mujoco.readthedocs.io/en/latest/mjx.html#feature-parity.\n
        """
        print(extended_msg)
        raise e


def load_mjx_model_and_data_from_file(
    filepath: Union[str, Path], force_float: bool = False
) -> Tuple[mjx.Model, mjx.Data]:
    """Convenience function for loading an mjx (model, data) pair from a filepath."""
    return mj_to_mjx_model_and_data(load_mj_model_from_file(filepath, force_float=force_float))
