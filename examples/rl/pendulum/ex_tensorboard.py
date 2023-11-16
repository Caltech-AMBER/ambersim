import functools
from datetime import datetime

import jax
from brax import envs
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from tensorboardX import SummaryWriter

from ambersim.rl.pendulum.swingup import PendulumSwingupEnv

"""
A pendulum swingup example that uses a tensorboard callback to log training
progress in real time.

You can view the logs with
$ tensorboard --logdir /tmp/ambersim
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

    # Set up a tensorboard logger
    log_dir = "/tmp/ambersim/pendulum"
    print(f"Setting up Tensorboard logging in {log_dir}")
    writer = SummaryWriter(log_dir)

    # Define a callback to log progress
    times = [datetime.now()]

    def progress(num_steps, metrics):
        """Logs progress during RL."""
        print(f"  Steps: {num_steps}, Reward: {metrics['eval/episode_reward']}")
        times.append(datetime.now())

        # Write all metrics to tensorboard
        for key, val in metrics.items():
            if isinstance(val, jax.Array):
                val = float(val)  # we need floats for logging
            writer.add_scalar(key, val, num_steps)

    # Do the training
    print("Training...")
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
    )

    print(f"Time to jit: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")
