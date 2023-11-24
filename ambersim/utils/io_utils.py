from pathlib import Path
from typing import Optional, Tuple, Union

import coacd
import mujoco as mj
import numpy as np
import trimesh
from dm_control import mjcf
from lxml import etree
from mujoco import mjx
from packaging import version

from ambersim import ROOT
from ambersim.utils._internal_utils import _check_filepath
from ambersim.utils.conversion_utils import save_model_xml


def set_actuators_type(
    model: mj.MjModel, actuator_type: str, actuator_idx: int, kp: float = 35, kd: float = 0
) -> mj.MjModel:
    """Modify the actuator type based on given input.

    Args:
        mj_model: A mujoco model
        actuator_type: ['position', 'torque']

    Returns:
        mj_model: A mujoco model.

    """
    # Make sure the actuator index is within the valid range
    if 0 <= actuator_idx < model.nu:
        # Use numpy to set the entire row to zeros
        model.actuator_gainprm[actuator_idx, :] = np.zeros_like(model.actuator_gainprm[actuator_idx, :])
        model.actuator_biasprm[actuator_idx, :] = np.zeros_like(model.actuator_biasprm[actuator_idx, :])

        # Make sure the actuator type is supported
        if actuator_type.lower() == "position":
            # Configure for position control
            model.actuator_gainprm[actuator_idx, 0] = kp  # Position gain
            model.actuator_biasprm[actuator_idx, 1] = -kp  # Position gain
            model.actuator_gainprm[actuator_idx, 2] = -kd  # Velocity gain (?)
        elif actuator_type.lower() == "torque":
            # Configure for torque control
            model.actuator_gainprm[actuator_idx, 0] = 1.0  # Torque gain

        else:
            raise ValueError("Actuator type must be one of: ['position', 'torque']!")
    else:
        raise IndexError("Actuator index is out of range!")

    return model


def _add_actuators(urdf_filepath: Union[str, Path], xml_filepath: Union[str, Path]) -> None:
    """Takes a URDF and a corresponding XML derived from it and adds actuators to the XML.

    This util is necessary because mujoco doesn't automatically add actuators. We assume that transmissions are defined
    for all actuated DOFs in the URDF. The supplied xml is directly modified.

    Args:
        urdf_filepath: A path to a URDF file.
        xml_filepath: A path to an XML file.
    """
    # parsing URDF
    # the recover=True keyword parses even after encountering invalid namespaces
    # this is useful when, e.g., we have drake:declare_convex, which is not a valid namespace
    with open(urdf_filepath, "r") as f:
        urdf_tree = etree.XML(f.read(), etree.XMLParser(remove_blank_text=True, recover=True))

    # parsing XML
    with open(xml_filepath, "r") as f:
        xml_tree = etree.XML(f.read(), etree.XMLParser(remove_blank_text=True))

    # checking whether the XML has an actuator section already
    if xml_tree.find("actuator") is None:
        xml_tree.append(etree.Element("actuator"))

    # getting transmission info from URDF and loading it into the XML
    actuators = xml_tree.find("actuator")
    for transmission in urdf_tree.findall("transmission"):
        joint_name = transmission.find("joint").get("name")
        joint = urdf_tree.find(f"joint[@name='{joint_name}']")
        limit = joint.find("limit").get("effort")

        if actuators.find(f"motor[@joint='{joint_name}']") is None:
            if limit is not None:
                etree.SubElement(
                    actuators,
                    "motor",
                    name=joint_name + "_actuator",
                    ctrllimited="true",
                    ctrlrange=f"-{limit} {limit}",
                    joint=joint_name,
                )
            else:
                etree.SubElement(
                    actuators,
                    "motor",
                    name=joint_name + "_actuator",
                    ctrllimited="false",
                    joint=joint_name,
                )

    # resaving the XML
    with open(xml_filepath, "wb") as f:
        f.write(etree.tostring(xml_tree, pretty_print=True))


def _add_mimics(urdf_filepath: Union[str, Path], xml_filepath: Union[str, Path]) -> None:
    """Takes a URDF and a corresponding XML derived from it and adds mimic joints to the XML.

    This util is necessary because mujoco doesn't automatically add equality constraints for mimic joints.

    Args:
        urdf_filepath: A path to a URDF file.
        xml_filepath: A path to an XML file.
    """
    # parsing URDF
    # the recover=True keyword parses even after encountering invalid namespaces
    # this is useful when, e.g., we have drake:declare_convex, which is not a valid namespace
    with open(urdf_filepath, "r") as f:
        urdf_tree = etree.XML(f.read(), etree.XMLParser(remove_blank_text=True, recover=True))

    # parsing XML
    with open(xml_filepath, "r") as f:
        xml_tree = etree.XML(f.read(), etree.XMLParser(remove_blank_text=True))

    # checking whether the XML has an equality section already
    if xml_tree.find("equality") is None:
        xml_tree.append(etree.Element("equality"))

    # getting mimic info from URDF and loading it into the XML
    equality = xml_tree.find("equality")
    for joint in urdf_tree.xpath("//joint[mimic]"):
        joint1 = joint.get("name")  # the joint that mimics
        mimic = joint.find("mimic")  # the mimic element
        joint2 = mimic.get("joint")  # the joint to mimic
        multiplier = mimic.get("multiplier") if mimic.get("multiplier") is not None else 1
        offset = mimic.get("offset") if mimic.get("offset") is not None else 0

        # adding the mimic joint
        etree.SubElement(
            equality,
            "joint",
            name=f"{joint1}_{joint2}_equality",
            joint1=joint1,
            joint2=joint2,
            polycoef=f"{offset} {multiplier} 0 0 0",
        )

    # resaving the XML
    with open(xml_filepath, "wb") as f:
        f.write(etree.tostring(xml_tree, pretty_print=True))


def _modify_robot_float_base(filepath: Union[str, Path]) -> mj.MjModel:
    """Modifies a robot to have a floating base if it doesn't already."""
    # loading current robot
    assert str(filepath).split(".")[-1] == "xml"
    robot = mjcf.from_file(filepath, model_dir="/".join(filepath.split("/")[:-1]))

    # only add free joint if the first body after worldbody has no freejoints
    assert robot.worldbody is not None
    if len(robot.worldbody.body[0].joint) == 0:
        robot.worldbody.body[0].add("freejoint", name="freejoint")
        assert robot.worldbody.body[0].inertial is not None  # checking for non-physical parsing errors

    # extracts the mujoco model from the dm_mujoco Physics object
    physics = mjcf.Physics.from_mjcf_model(robot)
    assert physics is not None  # pyright typechecking
    model = physics.model.ptr
    return model


def load_mj_model_from_file(
    filepath: Union[str, Path],
    force_float: bool = False,
    solver: Optional[Union[str, mj.mjtSolver]] = None,
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

    Raises:
        NotImplementedError: if the file extension is not in [".urdf", ".xml"]
    """
    # allow different solver specifications depending on mujoco version
    if version.parse(mj.__version__) < version.parse("3.0.1"):
        if solver is None:
            solver = mj.mjtSolver.mjSOL_CG
        elif isinstance(solver, str):
            if solver.lower() == "cg":
                solver = mj.mjtSolver.mjSOL_CG
            else:
                raise ValueError("Solver must be one of: ['cg']!")
        elif isinstance(solver, mj.mjtSolver):
            assert solver in [mj.mjtSolver.mjSOL_CG]
    else:
        if solver is None:
            solver = mj.mjtSolver.mjSOL_NEWTON
        elif isinstance(solver, str):
            if solver.lower() == "newton":
                solver = mj.mjtSolver.mjSOL_NEWTON
            elif solver.lower() == "cg":
                solver = mj.mjtSolver.mjSOL_CG
            else:
                raise ValueError("Solver must be one of: ['cg', 'newton']!")
        elif isinstance(solver, mj.mjtSolver):
            assert solver in [mj.mjtSolver.mjSOL_CG, mj.mjtSolver.mjSOL_NEWTON]

    filepath = _check_filepath(filepath)
    is_urdf = str(filepath).split(".")[-1] == "urdf"
    is_xml = str(filepath).split(".")[-1] == "xml"

    # treating URDFs and XMLs differently
    temp_output_path = False
    if is_urdf:
        output_path = "/".join(str(filepath).split("/")[:-1]) + "/_temp_xml_model.xml"
        save_model_xml(filepath, output_path=output_path)
        _add_actuators(filepath, output_path)
        _add_mimics(filepath, output_path)
        temp_output_path = True
    elif is_xml:
        output_path = filepath
    else:
        raise NotImplementedError

    # checking whether to force float
    if force_float:
        mj_model = _modify_robot_float_base(output_path)
    else:
        mj_model = mj.MjModel.from_xml_path(output_path)

    # deleting temp file
    if temp_output_path:
        Path(output_path).unlink()

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
