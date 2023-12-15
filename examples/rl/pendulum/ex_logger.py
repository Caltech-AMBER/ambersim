import functools
import os
from datetime import datetime

import jax
from brax import envs
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from ambersim.logger.logger import LoggerFactory
from ambersim.rl.pendulum.swingup import PendulumSwingupEnv

"""
A pendulum swingup example that uses a custom logger to log training
progress in real time.
"""

if __name__ == "__main__":
    # Initialize the environment
    envs.register_environment("pendulum_swingup", PendulumSwingupEnv)
    env = envs.get_environment("pendulum_swingup")

    # Define the training function
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(64,) * 3,
    )
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=100_000,
        num_evals=50,
        reward_scaling=0.1,
        episode_length=200,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=0,
        num_envs=1024,
        batch_size=512,
        network_factory=network_factory,
        seed=0,
    )

    # Save the log in the current directory
    log_dir = os.path.join(os.path.abspath(os.getcwd()), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print(f"Setting up Tensorboard logging in {log_dir}")
    logger = LoggerFactory.get_logger("tensorboard", log_dir)

    # Define a callback to log progress
    times = [datetime.now()]

    # Do the training
    print("Training...")
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=logger.log_progress,
    )
