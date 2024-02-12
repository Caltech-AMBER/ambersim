import os
from datetime import datetime
import functools

import jax
import yaml
from brax import envs
from brax.io import model

import wandb
from ambersim.envs.exo_base import Exo
from ambersim.utils import ppo_training_utils

# Set the GPU device to use (e.g., the first GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Disable memory preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"


def generate_policy_save_path(base_dir, env_name, policy_name_prefix):
    """Generates a save path for a policy."""
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Ensure the base directory exists
    os.makedirs(os.path.join(base_dir, env_name), exist_ok=True)

    # Generate the full path
    policy_name = f"{policy_name_prefix}_{timestamp}"
    return os.path.join(base_dir, env_name, policy_name)


def read_config(file_path):
    """Reads a yaml config file."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def merge_configs(*configs):
    """Merges multiple configs into one."""
    merged_config = {}
    for config in configs:
        merged_config.update(config)
    return merged_config


def struct_to_dict(struct_instance):
    """Converts a Flax struct dataclass instance to a dictionary."""
    return {field: getattr(struct_instance, field) for field in struct_instance.__annotations__}


env_name = "exo"
envs.register_environment("exo", Exo)
env = envs.get_environment(env_name)
home_dir = os.path.expanduser("~")
training_config = read_config(os.path.join(home_dir, "ambersim/models/exo/limits.yaml"))

# Add appo configurations
ppo_config = ppo_training_utils.PPOConfig()
object.__setattr__(ppo_config, 'num_timesteps', 50)
object.__setattr__(ppo_config, 'num_envs', 2)
object.__setattr__(ppo_config, 'episode_length', 50)
# ppo_config.num_envs = 2
training_config = merge_configs(training_config, struct_to_dict(ppo_config))
training_config = merge_configs(training_config, struct_to_dict(env.config))
# training_config.update(ppo_config)
# training_config.update(env.config)
# Initialize a wandb run
# wandb.init(project="exo_ppo", entity="kli5", config=training_config)
wandb.init(project="exo_ppo", config=training_config)

networks_factory = ppo_training_utils.make_networks_factory(ppo_config)
ppo_train_function = ppo_training_utils.train_fn(ppo_config)

times = [datetime.now()]


def progress(num_steps, metrics):
    """Logs progress during RL."""
    print(f"  Steps: {num_steps}, Reward: {metrics['eval/episode_reward']}")
    current_time = datetime.now()
    times.append(current_time)

    wandb_log_data = {key: float(val) if isinstance(val, jax.Array) else val for key, val in metrics.items()}
    wandb_log_data["current_time"] = current_time
    wandb.log(wandb_log_data, step=num_steps)


# Reset environments since internals may be overwritten by tracers from the
# domain randomization function.
env = envs.get_environment(env_name)
eval_env = envs.get_environment(env_name)
make_inference_fn, params, _ = ppo_train_function(environment=env, progress_fn=progress, eval_env=eval_env)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

base_dir = os.path.join(home_dir, "ambersim/policies")
policy_name_prefix = "ppo"
model_path = generate_policy_save_path(base_dir, env_name, policy_name_prefix)

# Save the model parameters
model.save_params(model_path, params)
print(f"Model saved at: {model_path}")
