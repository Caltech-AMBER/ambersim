# %%
import torch
torch.cuda.init()
import os
import time

import jax
import torch
import mediapy as media
import mujoco
from brax import envs
from mujoco import mjx
import numpy as np
import jax.numpy as jnp

from ambersim.envs.exo_base import Exo
from ambersim.utils import ppo_training_utils
from ambersim.utils.ahac import AHAC

# Set the GPU device to use (e.g., the first GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Disable memory preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["WANDB_DISABLED"] = "true"


# NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
# if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
#     with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
#         f.write(
#             """{
#     "file_format_version" : "1.0.0",
#     "ICD" : {
#         "library_path" : "libEGL_nvidia.so.0"
#     }
# }
# """
#         )

# Configure MuJoCo to use the EGL rendering backend (requires GPU)
print("Setting environment variable to use GPU rendering:")
os.environ["MUJOCO_GL"] = "egl"

# %%
# Usage
env_name = "exo"
envs.register_environment("exo", Exo)
env = envs.get_environment(env_name)
home_dir = os.path.expanduser("/work/exo/")
base_dir = os.path.join(home_dir, "ambersim/policies")

policy_name_prefix = "ppo"
# model_path = find_latest_policy(base_dir, env_name, policy_name_prefix)
model_path = "/work/exo/ambersim_cut/ahac_logs/init_policy.pt"
print("Latest policy:", model_path)

# ppo_config = ppo_training_utils.PPOConfig()
# networks_factory = ppo_training_utils.make_networks_factory(ppo_config)

# make_inference_fn, params = ppo_training_utils.load_model(
#     environment=env, network_factory=networks_factory, model_path=model_path
# )

# inference_fn = make_inference_fn(params)
agent = AHAC(env=env,
            actor_config={},
            critic_config={},
            steps_min=1000,  # minimum horizon
            steps_max=5000,  # maximum horizon
            max_epochs=5000,  # number of short rollouts to do (i.e. epochs)
            train=True,  # if False, we only eval the policy
            logdir="./ahac_logs",)
inference_fn = agent.load(model_path)
# jit_inference_fn = jax.jit(inference_fn)

eval_env = envs.get_environment(env_name)
eval_env.getRender()
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

rollout = []
actions = []
logged_data_per_step = []
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)

for i in range(5):
    start = time.time()
    rollout.append(state)
    act_rng, rng = jax.random.split(rng)
    with torch.no_grad(): 
        
        # print(jnp.array(state.info['obs_history']))
        # print(-agent.num_obs)
        # import ipdb; ipdb.set_trace()
        obs_tensor=torch.tensor(np.array(state.info['obs_history'][-agent.num_obs:]))
    # act in environment
    ctrl = jnp.array(agent.actor(obs_tensor.to(agent.device)).cpu().detach().numpy())
    # ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    #if state.done:
    #    state = jit_reset(rng)
    actions.append(ctrl)
    end = time.time()
    logged_data = env.log_state_info(state.info, ["domain_info", "tracking_err"], {})
    logged_data["tracking_foot_reward"] = state.info["reward_tuple"]["tracking_foot_reward"]
    logged_data_per_step.append(logged_data)
    if i % 100 == 0:
        print(f"step {i} time: {end - start}")

images = []
for i in range(len(rollout)):
    temp_State = rollout[i].pipeline_state
    images.append(eval_env.get_image(temp_State))

env.plot_logged_data(logged_data_per_step, save_dir="plots")

# media.show_video(images, fps=1.0 / eval_env.dt)
output_file = "video/exo_base_ppo_policy_new.mp4"
# Open the file in write mode and write content
with open(output_file, 'w') as file:
    media.write_video(output_file, images, fps=1.0 / eval_env.dt)

# Save the video
# breakpoint()
# %%


# from ambersim.utils.exo_sim_utils import plot_rewards, plot_rollout, plot_tracking_error

# plot_tracking_error(env,)
