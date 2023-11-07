from pathlib import Path

import mujoco as mj
from dm_control import mjcf
from mujoco import mjx

from ambersim import ROOT
from ambersim.utils.introspection_utils import get_joint_names
from ambersim.utils.io_utils import (
    _modify_robot_float_base,
    _rmtree,
    convex_decomposition_file,
    load_mjx_model_from_file,
    save_model_xml,
)


def test_load_model():
    """Tests model loading."""
    global_path = ROOT + "/models/pendulum/pendulum.urdf"
    local_path = "_test_local/pendulum.urdf"
    repo_path = "models/pendulum/pendulum.urdf"

    # creating temporary local file to test local paths
    local_dir = Path("_test_local")
    local_dir.mkdir(parents=True, exist_ok=False)
    with Path(local_path).open("w", encoding="utf-8") as f:
        f.write(Path(global_path).read_text())

    # string paths
    assert load_mjx_model_from_file(global_path)
    assert load_mjx_model_from_file(local_path)
    assert load_mjx_model_from_file(repo_path)

    # Path paths
    assert load_mjx_model_from_file(Path(global_path))
    assert load_mjx_model_from_file(Path(local_path))
    assert load_mjx_model_from_file(Path(repo_path))

    # remove temp local dir
    _rmtree(local_dir)


def test_save_xml():
    """Tests saving a URDF as an XML."""
    # saving a URDF as XML + verifying it loads into mjx
    save_model_xml(ROOT + "/models/pendulum/pendulum.urdf")
    assert load_mjx_model_from_file("pendulum.xml")
    Path.unlink("pendulum.xml")  # deleting test file


def test_force_float():
    """Tests the functionality of forcing models to have a floating base."""
    # case 1: model's first body has a joint already, so don't add a freejoint
    pend_path = ROOT + "/models/pendulum/pendulum.xml"
    model1 = _modify_robot_float_base(pend_path)
    combined1 = "\t".join(get_joint_names(model1))
    assert "freejoint" not in combined1

    # case 2: model's first body has no joints, so add a freejoint
    dummy_xml_string = """
    <mujoco model="parent">
      <worldbody>
        <body>
          <geom name="foo" type="box" pos="-0.2 0 0.3" size="0.5 0.3 0.1"/>
          <site name="attachment_site" pos="1. 2. 3." quat="1. 0. 0. 1."/>
          <body name="child" pos="1. 2. 3." quat="1. 0. 0. 1.">
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


def test_convex_decomposition():
    """Tests the convex decomposition util."""
    meshfile = "models/barrett_hand/meshes/finger.obj"
    savedir = Path("_test_dir")
    savedir.mkdir(parents=True, exist_ok=False)

    decomposed_meshes = convex_decomposition_file(meshfile, quiet=True, savedir=savedir)
    assert len(decomposed_meshes) > 0  # tests that there are meshes returned
    assert len(list(Path(savedir).glob("*.obj"))) > 0  # tests that the saving feature works
    _rmtree(savedir)
