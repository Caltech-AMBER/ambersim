# %%
import os
import time

import jax
import mediapy as media
import mujoco
from brax import envs
from mujoco import mjx

from ambersim.envs.exo_base import Exo
from ambersim.utils import ppo_training_utils

# Set the GPU device to use (e.g., the first GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Disable memory preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
    with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
        f.write(
            """{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
"""
        )

# Configure MuJoCo to use the EGL rendering backend (requires GPU)
print("Setting environment variable to use GPU rendering:")
os.environ["MUJOCO_GL"] = "egl"


def find_latest_policy(base_dir, env_name, policy_name_prefix):
    """Finds the latest policy file in a directory."""
    # Construct the path to the directory containing the policies
    directory = os.path.join(base_dir, env_name)

    # Ensure the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Filter and list all policy files
    policy_files = [
        file
        for file in os.listdir(directory)
        if file.startswith(policy_name_prefix) and os.path.isfile(os.path.join(directory, file))
    ]

    # Sort files by name in descending order
    policy_files.sort(reverse=True)

    # Return the first file in the list, which is the newest
    return os.path.join(directory, policy_files[0]) if policy_files else None


# %%
# Usage
env_name = "exo"
envs.register_environment("exo", Exo)
env = envs.get_environment(env_name)

home_dir = os.path.expanduser("~")

base_dir = os.path.join(home_dir, "ambersim/policies")

policy_name_prefix = "ppo"
model_path = find_latest_policy(base_dir, env_name, policy_name_prefix)
print("Latest policy:", model_path)

ppo_config = ppo_training_utils.PPOConfig()
networks_factory = ppo_training_utils.make_networks_factory(ppo_config)

make_inference_fn, params = ppo_training_utils.load_model(
    environment=env, network_factory=networks_factory, model_path=model_path
)

inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)


eval_env = envs.get_environment(env_name)
eval_env.getRender()
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

rollout = []
actions = []
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)

for _ in range(400):
    start = time.time()
    rollout.append(state)
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    actions.append(ctrl)
    end = time.time()
    print(f"step time: {end - start}")


images = []
for i in range(len(rollout)):
    temp_State = rollout[i].pipeline_state
    images.append(eval_env.get_image(temp_State))

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
