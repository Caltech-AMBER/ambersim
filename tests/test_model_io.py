from pathlib import Path

import igl
import mujoco as mj
import numpy as np
import trimesh
from dm_control import mjcf
from mujoco import mjx

from ambersim import ROOT
from ambersim.utils._internal_utils import _rmtree
from ambersim.utils.conversion_utils import convex_decomposition_file, save_model_xml
from ambersim.utils.introspection_utils import get_joint_names
from ambersim.utils.io_utils import _modify_robot_float_base, load_mjx_model_and_data_from_file


def test_load_model():
    """Tests model loading."""
    global_path = ROOT + "/models/pendulum/pendulum.urdf"
    local_path = "_test_local/pendulum.urdf"
    repo_path = "models/pendulum/pendulum.urdf"

    # creating temporary local file to test local paths
    local_dir = Path("_test_local")
    local_dir.mkdir(parents=True, exist_ok=True)
    with Path(local_path).open("w", encoding="utf-8") as f:
        f.write(Path(global_path).read_text())

    # string paths
    assert load_mjx_model_and_data_from_file(global_path)
    assert load_mjx_model_and_data_from_file(local_path)
    assert load_mjx_model_and_data_from_file(repo_path)

    # Path paths
    assert load_mjx_model_and_data_from_file(Path(global_path))
    assert load_mjx_model_and_data_from_file(Path(local_path))
    assert load_mjx_model_and_data_from_file(Path(repo_path))

    # remove temp local dir
    _rmtree(local_dir)


def test_all_models():
    """Tests the loading of all models in the repo."""
    filepaths = (p.resolve() for p in Path(ROOT + "/models").glob("**/*") if p.suffix in {".urdf", ".xml"})
    for filepath in filepaths:
        assert load_mjx_model_and_data_from_file(filepath)
        assert load_mjx_model_and_data_from_file(filepath, force_float=True)


def test_save_xml():
    """Tests saving a URDF as an XML."""
    # saving a URDF as XML + verifying it loads into mjx
    save_model_xml(ROOT + "/models/pendulum/pendulum.urdf")
    assert load_mjx_model_and_data_from_file("pendulum.xml")
    Path.unlink("pendulum.xml")  # deleting test file


def test_force_float():
    """Tests the functionality of forcing models to have a floating base."""
    # case 1: model's first body has a joint, add a freejoint with dummy body
    pend_path = ROOT + "/models/pendulum/pendulum.xml"
    model1 = _modify_robot_float_base(pend_path)
    combined1 = "\t".join(get_joint_names(model1))
    assert "freejoint" in combined1

    # case 2: model's first body has no joints, so add a freejoint
    dummy_xml_string = """
    <mujoco model="parent">
      <worldbody>
        <body>
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <geom name="foo" type="box" pos="-0.2 0 0.3" size="0.5 0.3 0.1"/>
          <body name="child" pos="1. 2. 3." quat="1. 0. 0. 1.">
            <inertial pos="0 0 0" mass="2" diaginertia="1 1 1"/>
            <geom name="my_box" type="box" pos="0.5 0.25 1." size="0.1 0.2 0.3"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    with open("_temp.xml", "w") as f:
        f.write(dummy_xml_string)

    model2 = _modify_robot_float_base("_temp.xml")
    Path.unlink("_temp.xml")
    combined2 = "\t".join(get_joint_names(model2))
    assert "freejoint" in combined2

    # case 3: add a freejoint to a URDF model with assets (much trickier b/c of file paths)
    _, data_unfree = load_mjx_model_and_data_from_file("models/barrett_hand/bh280.urdf", force_float=False)
    assert len(data_unfree.qpos) == 8  # 8 DOFs for the joints

    _, data_free = load_mjx_model_and_data_from_file("models/barrett_hand/bh280.urdf", force_float=True)
    assert len(data_free.qpos) == 15  # additional 7 quat states


def test_convex_decomposition():
    """Tests the convex decomposition util."""
    meshfile = "models/barrett_hand/meshes/finger.obj"
    savedir = Path("_test_dir")
    savedir.mkdir(parents=True, exist_ok=True)

    # tests that meshes are generated and saved correctly
    decomposed_meshes = convex_decomposition_file(meshfile, quiet=True, savedir=savedir)
    assert len(decomposed_meshes) > 0
    assert len(list(Path(savedir).glob("*.obj"))) > 0

    # tests that the decomposed meshes don't change compared to those in the mjx backend
    # in particular, checks that both trimesh and our decompositions have very close distance queries
    # see: https://github.com/google-deepmind/mujoco/blob/57e6940f579484adf34eebedc51279a818909f34/mjx/mujoco/mjx/_src/mesh.py#L195-L208
    for mesh in decomposed_meshes:
        # decomposed
        dverts = mesh.vertices
        dfaces = mesh.faces

        # trimesh convex hull
        tm = trimesh.Trimesh(vertices=dverts, faces=dfaces)
        tm_convex = trimesh.convex.convex_hull(tm)
        tverts = tm_convex.vertices
        tfaces = tm_convex.faces

        # checking that signed distance queries are functionally the same
        coords = np.random.randn(100, 3)
        signed_dist_d = igl.signed_distance(coords, dverts, dfaces)[0]
        signed_dist_t = igl.signed_distance(coords, tverts, tfaces)[0]
        assert np.allclose(signed_dist_d, signed_dist_t)

    _rmtree(savedir)
