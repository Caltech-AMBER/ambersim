import os

import jax

from ambersim.envs.exo_base import BehavState, Exo, ExoConfig
import jax.numpy as jp

config = ExoConfig()
# config.jt_traj_file = "medium_robust_flatfoot_gait.yaml"
# config.physics_steps_per_control_step = 1
config.residual_action_space = False
config.slope = True
config.hip_regulation = True
env = Exo(config)
filename = "video/base_slope_nominal.mp4"

# env.run_base_sim(rng=jax.random.PRNGKey(0), num_steps=400)
# env.run_base_bez_sim(rng=jax.random.PRNGKey(0), alpha = env.alpha,step_dur = env.step_dur, num_steps=400)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# env.run_base_sim(rng=jax.random.PRNGKey(0), alpha=env.alpha, num_steps=400, output_video=filename)
# TODO: jit, motion comfirm
state = env.reset(rng=jax.random.PRNGKey(0))
for i in range(100):
    env.step(state, action=jp.zeros(env.action_size))
# env.run_sim_from_standing(rng=jax.random.PRNGKey(0), num_steps=400)
# breakpoint()
# env.run_bez_sim_from_standing(
#     rng=jax.random.PRNGKey(0), alpha=env.alpha, step_dur=env.step_dur[BehavState.Walking], num_steps=400,output_video = filename
# )

# filename = "video/bez_policy_walking_nom.mp4"
# env.run_base_bez_sim(
#     rng=jax.random.PRNGKey(0),
#     alpha=env.alpha,
#     step_dur=env.step_dur[BehavState.Walking],
#     num_steps=400,
#     output_video=filename,
# )
