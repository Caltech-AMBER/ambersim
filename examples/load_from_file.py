import mujoco as mj
from mujoco import mjx

from ambersim import ROOT
from ambersim.utils.io_utils import load_mjx_model_and_data_from_file

"""This example demonstrates how to load robots from URDFs/XMLs into mujoco/mjx.

All of the following work:
(1) a global path (ROOT specifies the repository root globally)
(2) a local path
(3) a path specified with respect to the repository root

Additionally, the paths can be passed as strings or Path objects.
"""

mjx_model1, mjx_data1 = load_mjx_model_and_data_from_file(ROOT + "/models/pendulum/pendulum.urdf")  # (1)
mjx_model3, mjx_data3 = load_mjx_model_and_data_from_file("models/pendulum/pendulum.urdf")  # (3)
