import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from jax import jit

from ambersim.control.predictive_control import PDPredictiveSamplingController, PDPredictiveSamplingControllerParams
from ambersim.trajopt.cost import StaticGoalQuadraticCost
from ambersim.trajopt.shooting import PDPredictiveSampler
from ambersim.utils.io_utils import load_mj_model_from_file, mj_to_mjx_model_and_data

"""
By far the most sensitive parameters for this controller parameterization are the rollout horizon N and the proportional gain.
"""

# ########## #
# PARAMETERS #
# ########## #

dt = 0.001  # rate of reality
num_phys_steps_per_control_step = 15  # dt * npspcs = rate of control

w_pos = 10.0
w_vel = 0.01
w_ctrl = 0.001
q_goal = jnp.array([0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 1.0, 0.8, 0.5, 0.5])

nsamples = 100  # number of samples to draw for predictive sampling
stdev = 0.1  # standard deviation of control parameters to draw
N = 8  # number of time steps to predict

key = jax.random.PRNGKey(1234)

# ########## #
# SIMULATION #
# ########## #

# loading model
mj_model = load_mj_model_from_file("models/allegro_hand/right_hand_motor.xml")
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
    xg=jnp.concatenate((q_goal, jnp.zeros(16))),
)
ps = PDPredictiveSampler(model=ctrl_model, cost_function=cost_function, nsamples=nsamples, stdev=stdev)

kp = 5.0
kd = 0.1
controller = PDPredictiveSamplingController(trajectory_optimizer=ps, model=ctrl_model)
jit_compute = jit(
    lambda key, x_meas, qgs_guess: controller.compute_with_qs_star(
        PDPredictiveSamplingControllerParams(key=key, x=x_meas, guess=qgs_guess, kp=kp, kd=kd)
    )
)
print("Controller created! Simulating...")

# simulating forward
x0 = 2.0 * (jax.random.uniform(key, shape=(ctrl_model.nq + ctrl_model.nv,)) - 0.5)  # random initial state

mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = x0[: mj_model.nq]
mj_data.qvel[:] = x0[mj_model.nq :]

qgs_guess = jnp.tile(x0[: mj_model.nq], (N, 1))

print("Compiling jit_compute...")
jit_compute(key, x0, qgs_guess)
print("Compiled! Beginning simulation...")

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        while True:
            start = time.time()
            x_meas = jnp.concatenate((mj_data.qpos, mj_data.qvel))
            qg, _qgs_guess = jit_compute(key, x_meas, qgs_guess)
            qg.block_until_ready()
            qgs_guess = _qgs_guess[1:, :]
            print(f"Controller delay: {time.time() - start}")
            for _ in range(num_phys_steps_per_control_step):
                start = time.time()
                u = -kp * (mj_data.qpos - qg) - kd * mj_data.qvel
                u = np.clip(u, mj_model.actuator_ctrlrange[:, 0], mj_model.actuator_ctrlrange[:, 1])
                mj_data.ctrl[:] = u
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
                elapsed = time.time() - start
                if elapsed < mj_model.opt.timestep:
                    time.sleep(mj_model.opt.timestep - elapsed)
            key = jax.random.split(key)[0]
