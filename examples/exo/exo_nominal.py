import os

import jax

from ambersim.envs.exo_base import BehavState, Exo, ExoConfig

config = ExoConfig()
# config.physics_steps_per_control_step = 1
env = Exo(config)
# filename = "bez_policy_from_standing_video_0.001.mp4"

# env.run_base_sim(rng=jax.random.PRNGKey(0), num_steps=400)
# env.run_base_bez_sim(rng=jax.random.PRNGKey(0), alpha = env.alpha,step_dur = env.step_dur, num_steps=400)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# env.run_base_sim(rng=jax.random.PRNGKey(0), num_steps=400)
# env.run_sim_from_standing(rng=jax.random.PRNGKey(0), num_steps=400)
# breakpoint()
# env.run_bez_sim_from_standing(
#     rng=jax.random.PRNGKey(0), alpha=env.alpha, step_dur=env.step_dur[BehavState.Walking], num_steps=400,output_video = filename
# )
filename = "bez_policy_walking.mp4"
env.run_base_bez_sim(
    rng=jax.random.PRNGKey(0),
    alpha=env.alpha,
    step_dur=env.step_dur[BehavState.Walking],
    num_steps=400,
    output_video=filename,
)
