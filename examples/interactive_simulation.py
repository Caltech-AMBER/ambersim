import time

import mujoco
import mujoco.viewer
import numpy as np

from ambersim.utils.io_utils import load_mj_model_from_file

"""Example of launching an interactive simulation with a custom controller in
mujoco. The controller keeps running, while the user can interact with the
system through the GUI.
"""


def swingup_controller(theta: float, theta_dot: float) -> float:
    """A simple swingup controller for the pendulum.

    Adopted from https://underactuated.mit.edu/pend.html#section3.

    Args:
        theta: The angle of the pendulum, in radians (zero is down).
        theta_dot: Velocity, in radians per second.

    Returns:
        tau: The control torque to apply.
    """
    # Approximate model: note that this is not an exact match for the sim.
    m = 1.0  # mass
    l = 0.7  # length
    g = 9.81  # gravity

    if np.cos(theta) > -0.9:
        # Use an energy shaping controller to pump the pendulum up
        desired_energy = m * g * l
        energy = 0.5 * m * l**2 * theta_dot**2 - m * g * l * np.cos(theta)
        k = 0.1
        tau = -k * theta_dot * (energy - desired_energy)
    else:
        # Switch to a PD controller near the top
        kp = 10.0
        kd = 5.0
        theta_err = np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi))
        tau = -kp * theta_err - kd * theta_dot

    return tau


if __name__ == "__main__":
    # Create the model
    mj_model = load_mj_model_from_file("models/pendulum/scene.xml")
    mj_data = mujoco.MjData(mj_model)

    # Set the initial state
    mj_data.qpos[:] = 0.01

    # Launch the viewer
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Compute the control signal
            theta = mj_data.qpos[0]
            theta_dot = mj_data.qvel[0]
            tau = swingup_controller(theta, theta_dot)
            mj_data.ctrl[0] = tau

            # Step the simulation forward
            mujoco.mj_step(mj_model, mj_data)

            # Sync data from the viewer with the simulation
            viewer.sync()

            # Try to run in roughly real time
            elapsed = time.time() - step_start
            if elapsed < mj_model.opt.timestep:
                time.sleep(mj_model.opt.timestep - elapsed)
