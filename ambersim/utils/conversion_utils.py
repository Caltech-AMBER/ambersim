from pathlib import Path
from typing import List, Optional, Union

import coacd
import mujoco as mj
import trimesh

from ambersim.utils._internal_utils import _check_filepath


def save_model_xml(filepath: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> None:
    """Loads a model and saves it to a mujoco-compliant XML.

    Will save the file to the directory where this util is called. Note that you should add a mujoco tag to URDF files
    to specify things like the mesh directory relative to the URDF. See the below link for details:
    https://mujoco.readthedocs.io/en/latest/modeling.html?highlight=urdf#urdf-extensions

    Args:
        filepath: A path to a URDF or MJCF file. This can be global, local, or with respect to the repository root.
        output_path: The output path of the model.
    """
    try:
        # loading model and saving XML
        filepath = _check_filepath(filepath)
        _model = mj.MjModel.from_xml_path(filepath)
        if output_path is None:
            output_path = filepath.split("/")[-1].split(".")[0] + ".xml"
        mj.mj_saveLastXML(output_path, _model)

        # reporting save path for clarity
        print(f"XML file saved to {str(output_path)}!")
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
