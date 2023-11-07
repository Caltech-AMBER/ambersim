from pathlib import Path
from typing import List, Optional, Tuple, Union

import coacd
import mujoco as mj
import trimesh
from dm_control import mjcf
from mujoco import mjx

from ambersim import ROOT

# ############## #
# INTERNAL UTILS #
# ############## #


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


def _modify_robot_float_base(filepath: Union[str, Path]) -> mj.MjModel:
    """Modifies a robot to have a floating base if it doesn't already."""
    # loading current robot
    assert str(filepath).split(".")[-1] == "xml"
    robot = mjcf.from_file(filepath)

    # only add free joint if the first body after worldbody has no joints
    assert robot.worldbody is not None
    if len(robot.worldbody.body[0].joint) == 0:
        arena = mjcf.RootElement(model=robot.model)
        attachment_frame = arena.attach(robot)
        attachment_frame.add("freejoint", name="freejoint")
        robot = arena
    model = mj.MjModel.from_xml_string(robot.to_xml_string())
    return model


# ############# #
# MODEL LOADING #
# ############# #


def load_mjx_model_from_file(filepath: Union[str, Path], force_float: bool = False) -> Tuple[mjx.Model, mjx.Data]:
    """Loads a mjx model/data pair from a filepath.

    Args:
        filepath: A path to a URDF or MJCF file. This can be global, local, or with respect to the repository root.
        force_float: Whether to forcibly float the base of the robot.

    Returns:
        mjx_model: A mjx Model.
        mjx_data: A mjx Data struct.
    """
    filepath = _check_filepath(filepath)

    # loading the model and data. check whether freejoint is added forcibly
    if force_float:
        # check that the file is an XML. if not, save as xml temporarily
        if str(filepath).split(".")[-1] != "xml":
            save_model_xml(filepath, output_name="_temp_xml_model")
            _model = _modify_robot_float_base("_temp_xml_model.xml")
            Path.unlink("_temp_xml_model.xml")
        else:
            _model = _modify_robot_float_base(filepath)
    else:
        _model = mj.MjModel.from_xml_path(filepath)

    try:
        mjx_model = mjx.device_put(_model)
        mjx_data = mjx.make_data(_model)
    except NotImplementedError as e:
        print()
        print("There are some URDF convex primitives that aren't compatible with mjx's convex collision checking.")
        print(
            "See: https://github.com/google-deepmind/mujoco/blob/57e6940f579484adf34eebedc51279a818909f34/mjx/mujoco/mjx/_src/collision_driver.py#L47-L62"
        )
        print()
        raise e

    return mjx_model, mjx_data


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


def convex_decomposition_file(meshfile: Union[str, Path], quiet: bool = True, **kwargs) -> List[trimesh.Trimesh]:
    """Performs a convex decomposition on a mesh using coacd.

    For a description of all the kwargs you can pass, see https://github.com/SarahWeiii/CoACD/blob/main/python/package/bin/coacd

    Args:
        meshfile: A path to a URDF or MJCF file. This can be global, local, or with respect to the repository root.

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
    mesh = coacd.Mesh(_mesh.vertices, _mesh.faces)
    parts = coacd.run_coacd(mesh, **kwargs)  # list of (vert, face) tuples
    decomposed_meshes = [trimesh.Trimesh(vertices=verts, faces=faces) for (verts, faces) in parts]
    return decomposed_meshes


def convex_decomposition_dir(
    meshdir: Union[str, Path],
    quiet: bool = False,
    savedir: Optional[Union[str, Path]] = None,
) -> List[List[trimesh.Trimesh]]:
    """Performs convex decompositions on all meshes (recursively) in a specified directory.

    Args:
        meshdir: A path to dir containing meshes. This can be global, local, or with respect to the repository root.
        quiet: Whether coacd output should be suppressed.
        savedir: If supplied, where to save the output meshes.

    Returns:
        all_decomposed_meshes: A list of lists of Trimesh objects representing the convex decompositions.
    """
    meshdir = _check_filepath(meshdir)
    all_decomposed_meshes = []

    # coacd only works on .obj files, so we only search for those (recursively) in meshdir
    for meshfile in Path(meshdir).rglob("*.obj"):
        decomposed_meshes = convex_decomposition_file(meshfile, quiet=quiet)
        if savedir is not None:
            name = str(meshfile).split("/")[-1].split(".")[0]
            for i, mesh in enumerate(decomposed_meshes):
                (Path(savedir) / Path(name)).mkdir(parents=True, exist_ok=True)  # make subdir for each decomposed mesh
                mesh.export(Path(savedir) / Path(name + f"/col_{i}.obj"))
        all_decomposed_meshes.append(decomposed_meshes)
    return all_decomposed_meshes


# ############# #
# INTROSPECTION #
# ############# #


def get_geom_names(model: mj.MjModel) -> List[str]:
    """Returns a list of all geom names in a mujoco (NOT mjx) model."""
    return [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]


def get_joint_names(model: mj.MjModel) -> List[str]:
    """Returns a list of all joint names in a mujoco (NOT mjx) model."""
    return [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
