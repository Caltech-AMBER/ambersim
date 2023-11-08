from pathlib import Path
from typing import List, Optional, Tuple, Union

import coacd
import mujoco as mj
import numpy as np
import trimesh
from dm_control import mjcf
from mujoco import mjx

from ambersim import ROOT


def _check_filepath(filepath: Union[str, Path]) -> str:
    """Checks validity of a filepath for model loading."""
    assert isinstance(filepath, (str, Path))

    # checking whether file exists
    if isinstance(filepath, str):
        filepath = Path(filepath)  # global/local
    if not filepath.exists():
        filepath = ROOT / filepath  # repo root
        if not filepath.exists():
            raise ValueError("The model file doesn't exist at the specified path!")
    filepath = str(filepath)
    return filepath


def _rmtree(f: Path):
    """Recursively deletes a directory using pathlib.

    See: https://stackoverflow.com/a/66552066
    """
    if f.is_file():
        f.unlink()
    else:
        for child in f.iterdir():
            _rmtree(child)
        f.rmdir()


# ############# #
# MODEL LOADING #
# ############# #


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

        # ensuring that the base link has inertial attributes
        assert robot.worldbody is not None  # pyright typechecking
        if robot.worldbody.body[0].inertial is None:
            robot.worldbody.body[0].add("inertial", mass=1e-6, diaginertia=1e-6 * np.ones(3), pos=np.zeros(3))

    # extracts the mujoco model from the dm_mujoco Physics object
    physics = mjcf.Physics.from_mjcf_model(robot)
    assert physics is not None  # pyright typechecking
    model = physics.model.ptr
    return model


def load_mj_model_from_file(filepath: Union[str, Path], force_float: bool = False) -> mj.MjModel:
    """Loads a mujoco model from a filepath.

    Args:
        filepath: A path to a URDF or MJCF file. This can be global, local, or with respect to the repository root.
        force_float: Whether to forcibly float the base of the robot.

    Returns:
        mj_model: A mujoco model.
    """
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


# ################ #
# MODEL CONVERSION #
# ################ #


def save_model_xml(filepath: Union[str, Path], output_name: Optional[str] = None) -> None:
    """Loads a model and saves it to a mujoco-compliant XML.

    Will save the file to the directory where this util is called. Note that you should add a mujoco tag to URDF files
    to specify things like the mesh directory relative to the URDF. See the below link for details:
    https://mujoco.readthedocs.io/en/latest/modeling.html?highlight=urdf#urdf-extensions

    Args:
        filepath: A path to a URDF or MJCF file. This can be global, local, or with respect to the repository root.
        output_name: The output name of the model.
    """
    try:
        # loading model and saving XML
        filepath = _check_filepath(filepath)
        _model = mj.MjModel.from_xml_path(filepath)
        if output_name is None:
            output_name = filepath.split("/")[-1].split(".")[0]
        else:
            output_name = output_name.split(".")[0]  # strip any extensions
        mj.mj_saveLastXML(f"{output_name}.xml", _model)

        # reporting save path for clarity
        output_path = Path.cwd() / Path(str(output_name) + ".xml")
        print(f"XML file saved to {output_path}!")
    except ValueError as e:
        print(e)
        print(
            "If you're getting errors about mesh filepaths, "
            + "make sure to add a mujoco tag to the URDF to specify the meshdir!"
        )


def convex_decomposition_file(
    meshfile: Union[str, Path],
    quiet: bool = False,
    savedir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> List[trimesh.Trimesh]:
    """Performs a convex decomposition on a mesh using coacd.

    For a description of all the kwargs you can pass, see https://github.com/SarahWeiii/CoACD/blob/main/python/package/bin/coacd

    Args:
        meshfile: A path to a URDF or MJCF file. This can be global, local, or with respect to the repository root.
        quiet: Whether to suppress coacd output.
        savedir: If supplied, where to save the output meshes.

    Returns:
        decomposed_meshes: A list of Trimesh objects forming the convex decomposition.
    """
    # checking defaults for keyword args
    if "max_convex_hull" not in kwargs:
        kwargs["max_convex_hull"] = 16
    if "threshold" not in kwargs:
        kwargs["threshold"] = 0.1

    # turn off verbose outputs
    if quiet:
        coacd.set_log_level("error")

    # executing the convex decomposition
    meshfile = _check_filepath(meshfile)
    _mesh = trimesh.load(meshfile, force="mesh")
    full_mesh = coacd.Mesh(_mesh.vertices, _mesh.faces)
    parts = coacd.run_coacd(full_mesh, **kwargs)  # list of (vert, face) tuples
    decomposed_meshes = [trimesh.Trimesh(vertices=verts, faces=faces) for (verts, faces) in parts]

    # saving decomposed meshes
    if savedir is not None:
        name = str(meshfile).split("/")[-1].split(".")[0]
        for i, mesh in enumerate(decomposed_meshes):
            mesh.export(Path(savedir) / Path(name + f"_col_{i}.obj"))

    return decomposed_meshes


def convex_decomposition_dir(
    meshdir: Union[str, Path],
    recursive: bool = False,
    quiet: bool = False,
    savedir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> List[List[trimesh.Trimesh]]:
    """Performs convex decompositions on all meshes in a specified directory.

    Args:
        meshdir: A path to dir containing meshes. This can be global, local, or with respect to the repository root.
        recursive: Whether to recursively search for mesh files.
        quiet: Whether to suppress coacd output.
        savedir: If supplied, where to save the output meshes.

    Returns:
        all_decomposed_meshes: A list of lists of Trimesh objects representing the convex decompositions.
    """
    meshdir = _check_filepath(meshdir)
    if recursive:
        glob_func = Path(meshdir).rglob
    else:
        glob_func = Path(meshdir).glob
    all_decomposed_meshes = []

    # coacd only works on .obj files, so we only search for those (recursively) in meshdir
    for meshfile in glob_func("*.obj"):
        decomposed_meshes = convex_decomposition_file(meshfile, quiet=quiet, savedir=savedir, **kwargs)
        all_decomposed_meshes.append(decomposed_meshes)
    return all_decomposed_meshes
