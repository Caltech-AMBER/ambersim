import os

import jax

from ambersim.envs.exo_base import BehavState, Exo, ExoConfig
import jax.numpy as jp
import mediapy as media

print("Initializing...")
config = ExoConfig()
# config.jt_traj_file = "medium_robust_flatfoot_gait.yaml"
# config.physics_steps_per_control_step = 1
config.residual_action_space = False
config.slope = True
config.hip_regulation = True
env = Exo(config)

# env.run_base_sim(rng=jax.random.PRNGKey(0), num_steps=400)
# env.run_base_bez_sim(rng=jax.random.PRNGKey(0), alpha = env.alpha,step_dur = env.step_dur, num_steps=400)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# env.run_base_sim(rng=jax.random.PRNGKey(0), alpha=env.alpha, num_steps=400, output_video=filename)
# TODO: jit, motion comfirm
state = env.reset(rng=jax.random.PRNGKey(0))
env.step(state, jp.zeros(env.action_size))

jit_env_step = jax.jit(env.step)

output_video = "nominal_policy_video_ibs.mp4"
num_steps = 500
"""Run the simulation basic version."""
env.getRender()

images = []
logged_data_per_step = []
print("Starting simulation...")
for _ in range(num_steps):
    state = jit_env_step(state, jp.zeros(env.action_size))
    images.append(env.get_image(state.pipeline_state))
    logged_data = {}
    logged_data = env.log_state_info(
        state.info, ["domain_info", "tracking_err", "joint_desire", "reward_tuple", "blended_action"], logged_data
    )
    logged_data["tracking_foot_reward"] = state.info["reward_tuple"]["tracking_foot_reward"]
    logged_data_per_step.append(logged_data)

media.write_video(output_video, images, fps=1.0 / env.dt)

env.plot_logged_data(logged_data_per_step, save_dir="plots")
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
