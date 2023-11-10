from pathlib import Path

from ambersim import ROOT
from ambersim.utils._internal_utils import _rmtree
from ambersim.utils.conversion_utils import convex_decomposition_dir, convex_decomposition_file

"""This example demonstrates how to perform convex decompositions of nonconvex meshes.

We use coacd under the hood and have functionality to process both single files and directories.
"""

# directories of interest
barrett_hand_mesh_dir = Path(ROOT + "/models/barrett_hand/meshes")  # directory with a lot of meshes
test_model_dir = Path(ROOT + "/_test_model_dir")  # just for demonstration
test_save_dir = Path(ROOT + "/_test_output_dir")  # where decomposed meshes will be saved

# example 1: single file
test_save_dir.mkdir(parents=True, exist_ok=True)
decomposed_meshes = convex_decomposition_file(
    barrett_hand_mesh_dir / Path("finger.obj"),  # decomposing the finger geometry
    quiet=True,  # suppress coacd output
    savedir=test_save_dir,  # save the output meshes into the specified directory
)
print("Example 1: paths of decomposed files:")
for f in test_save_dir.glob("*.obj"):
    print(str(f))
_rmtree(test_save_dir)  # remove the test directory (delete this if you want to keep saved files)

# example 2: whole directory
# setting up dummy model directory with two meshes in it
test_save_dir.mkdir(parents=True, exist_ok=True)
test_model_dir.mkdir(parents=True, exist_ok=True)
with open(test_model_dir / Path("finger.obj"), "a") as f:
    f.write((barrett_hand_mesh_dir / Path("finger.obj")).read_text())
with open(test_model_dir / Path("finger_tip.obj"), "a") as f:
    f.write((barrett_hand_mesh_dir / Path("finger_tip.obj")).read_text())

decomposed_meshes = convex_decomposition_dir(
    test_model_dir,
    recursive=False,  # whether to recurse in the specified directory
    quiet=True,
    savedir=test_save_dir,
)
print("Example 2: paths of decomposed files:")
for f in test_save_dir.glob("*.obj"):
    print(str(f))
_rmtree(test_model_dir)
_rmtree(test_save_dir)
