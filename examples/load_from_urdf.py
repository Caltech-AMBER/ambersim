import mujoco as mj
from mujoco import mjx

from ambersim import ROOT

"""This example demonstrates how to load robots from URDFs into mujoco/mjx."""

# vanilla mujoco model (NOT jax-compatible)
_model = mj.MjModel.from_xml_path(ROOT + "/models/pendulum/pendulum.urdf")

# jax-compatible mjx models
mjx_model = mjx.device_put(_model)
mjx_data = mjx.make_data(_model)
