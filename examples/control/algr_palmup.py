import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from jax import jit

# from ambersim.control.predictive_control import PDPredictiveSamplingController, PDPredictiveSamplingControllerParams
from ambersim.control.predictive_control import (
    VanillaPredictiveSamplingController,
    VanillaPredictiveSamplingControllerParams,
)
from ambersim.trajopt.cost import StaticGoalQuadraticCost

# from ambersim.trajopt.shooting import PDPredictiveSampler
from ambersim.trajopt.shooting import VanillaPredictiveSampler
from ambersim.utils.io_utils import load_mj_model_from_file, mj_to_mjx_model_and_data

"""Important info:
* the cube's states come after the hand's states
* the allegro hand is floating but fixed in space

TODOs:
* implement a check for the cube being stuck
* add some reasonable cost terms for the palm up task (see the mjpc code + talk to vince)
* add support for the PD controller, but we have to more clever about underactuation
"""

# ##################### #
# CUBE TARGET POSITIONS #
# ##################### #

# position
POS = jnp.array([0.05, 0.0, 0.05])  # move the cube over the fingers
Z_FAIL = -0.05  # floor at -0.1

# rotations
IDENTITY = jnp.array([1.0, 0.0, 0.0, 0.0])
X90 = jnp.array([0.7071068, 0.7071068, 0.0, 0.0])
X180 = jnp.array([0.0, 1.0, 0.0, 0.0])
X270 = jnp.array([-0.7071068, 0.7071068, 0.0, 0.0])
Y90 = jnp.array([0.7071068, 0.0, 0.7071068, 0.0])
Y180 = jnp.array([0.0, 0.0, 1.0, 0.0])
Y270 = jnp.array([-0.7071068, 0.0, 0.7071068, 0.0])
Z90 = jnp.array([0.7071068, 0.0, 0.0, 0.7071068])
Z180 = jnp.array([0.0, 0.0, 0.0, 1.0])
Z270 = jnp.array([-0.7071068, 0.0, 0.0, 0.7071068])

# ########## #
# PARAMETERS #
# ########## #

dt = 0.001  # rate of reality
num_phys_steps_per_control_step = 15  # dt * npspcs = rate of control

w_pos = 10.0
w_vel = 0.001
w_ctrl = 0.001
q_goal_algr = jnp.zeros(16)
q_goal_cube = jnp.concatenate((POS, X90))
q_goal = jnp.concatenate((q_goal_algr, q_goal_cube))

nsamples = 100  # number of samples to draw for predictive sampling
stdev = 0.3  # standard deviation of control parameters to draw
N = 5  # number of time steps to predict

key = jax.random.PRNGKey(1234)

# ########## #
# SIMULATION #
# ########## #

# loading model
mj_model = load_mj_model_from_file("models/allegro_hand/scene_right.xml")
mj_model.opt.timestep = dt

# defining the predictive controller
ctrl_model, _ = mj_to_mjx_model_and_data(mj_model)
ctrl_model = ctrl_model.replace(
    opt=ctrl_model.opt.replace(
        timestep=num_phys_steps_per_control_step * dt,  # dt for each step in the controller's internal model
        iterations=1,  # number of Newton steps to take during solve
        ls_iterations=1,  # number of line search iterations along step direction
        integrator=0,  # Euler semi-implicit integration
        solver=2,  # Newton solver
    )
)
cost_function = StaticGoalQuadraticCost(
    Q=jnp.diag(jnp.array([w_pos] * ctrl_model.nq + [w_vel] * ctrl_model.nv)),
    Qf=jnp.diag(jnp.array([w_pos] * ctrl_model.nq + [w_vel] * ctrl_model.nv)),
    R=w_ctrl * jnp.eye(ctrl_model.nu),
    xg=jnp.concatenate((q_goal, jnp.zeros(ctrl_model.nv))),
)
ps = VanillaPredictiveSampler(model=ctrl_model, cost_function=cost_function, nsamples=nsamples, stdev=stdev)
controller = VanillaPredictiveSamplingController(trajectory_optimizer=ps, model=ctrl_model)
jit_compute = jit(
    lambda key, x_meas, us_guess: controller.compute_with_us_star(
        VanillaPredictiveSamplingControllerParams(key=key, x=x_meas, guess=us_guess)
    )
)
print("Controller created! Simulating...")

# simulating forward
q0_algr = jnp.zeros(16)
q0_cube = jnp.concatenate((POS, IDENTITY))
v0 = jnp.zeros(mj_model.nv)
x0 = jnp.concatenate((q0_algr, q0_cube, v0))

mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = x0[: mj_model.nq]
mj_data.qvel[:] = x0[mj_model.nq :]

us_guess = jnp.zeros((N, mj_model.nu))

print("Compiling jit_compute...")
jit_compute(key, x0, us_guess)
print("Compiled! Beginning simulation...")

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        while True:
            start = time.time()
            x_meas = jnp.concatenate((mj_data.qpos, mj_data.qvel))
            u, _us_guess = jit_compute(key, x_meas, us_guess)
            us_guess = np.concatenate((_us_guess[1:, :], np.zeros((1, mj_model.nu))))
            mj_data.ctrl[:] = u
            print(f"Controller delay: {time.time() - start}")
            for _ in range(num_phys_steps_per_control_step):
                start = time.time()
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
                elapsed = time.time() - start
                if elapsed < mj_model.opt.timestep:
                    time.sleep(mj_model.opt.timestep - elapsed)

            # check whether to reset the cube
            if mj_data.qpos[-5] <= Z_FAIL:
                print("*** Cube fell! Resetting... *** ")
                mj_data.qpos[:] = x0[: mj_model.nq]
                mj_data.qvel[:] = x0[mj_model.nq :]
                us_guess = jnp.zeros((N, mj_model.nu))
                viewer.sync()
                time.sleep(1.0)

            key = jax.random.split(key)[0]
