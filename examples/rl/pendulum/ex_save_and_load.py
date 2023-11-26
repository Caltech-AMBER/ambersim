import functools
import pickle
import sys
import time
from datetime import datetime

import jax
import mujoco
import mujoco.viewer
from brax import envs
from brax.io import model
from brax.training import distribution
from brax.training.acme import running_statistics
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo.networks import make_inference_fn
from mujoco import mjx

from ambersim.learning.architectures import MLP
from ambersim.rl.helpers import BraxPPONetworksWrapper
from ambersim.rl.pendulum.swingup import PendulumSwingupEnv

"""
Perform pendulum swingup training with custom policy architectures, and
demonstrate saving and loading the trained networks from disk.
"""


def train_swingup():
    """Train a pendulum swingup agent with custom network architectures."""
    # Initialize the environment
    envs.register_environment("pendulum_swingup", PendulumSwingupEnv)
    env = envs.get_environment("pendulum_swingup")

    # Create some funky custom network architectures
    policy_network = MLP(layer_sizes=(512, 4, 128, 2))  # outputs mean and variance
    value_network = MLP(layer_sizes=(26, 128, 53, 1))  # outputs a single value
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=policy_network,
        value_network=value_network,
        action_distribution=distribution.NormalTanhDistribution,
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
        network_factory=network_wrapper.make_ppo_networks,
        seed=0,
    )

    # Define a callback to log progress
    times = [datetime.now()]

    def progress(num_steps, metrics):
        """Logs progress during RL."""
        print(f"  Steps: {num_steps}, Reward: {metrics['eval/episode_reward']}")
        times.append(datetime.now())

    # Do the training
    print("Training...")
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
    )

    print(f"Time to jit: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

    # Save both the parameters and the networks to disk
    print("Saving...")
    params_path = "/tmp/pendulum_params.pkl"
    networks_path = "/tmp/pendulum_networks.pkl"
    model.save_params(params_path, params)
    with open(networks_path, "wb") as f:
        pickle.dump(network_wrapper, f)


def test_trained_swingup_policy():
    """Load a trained policy and run an interactive simulation."""
    envs.register_environment("pendulum_swingup", PendulumSwingupEnv)
    env = envs.get_environment("pendulum_swingup")
    mj_model = env.model
    mj_data = mujoco.MjData(mj_model)
    obs = env.compute_obs(mjx.device_put(mj_data), {})

    print("Loading trained policy...")
    params_path = "/tmp/pendulum_params.pkl"
    networks_path = "/tmp/pendulum_networks.pkl"
    params = model.load_params(params_path)
    with open(networks_path, "rb") as f:
        network_wrapper = pickle.load(f)

    # Create the policy
    ppo_networks = network_wrapper.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )

    make_policy = make_inference_fn(ppo_networks)
    policy = make_policy(params, deterministic=True)
    jit_policy = jax.jit(policy)

    print("Simulating...")
    rng = jax.random.PRNGKey(0)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            start_time = time.time()
            act_rng, rng = jax.random.split(rng)

            # Apply the policy
            act, _ = jit_policy(obs, act_rng)
            mj_data.ctrl[:] = act
            obs = env.compute_obs(mjx.device_put(mj_data), {})

            # Step the simulation
            for _ in range(env._physics_steps_per_control_step):
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            dt = float(env.dt)
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    usage_message = "Usage: python ex_save_and_load.py [train|test]"

    if len(sys.argv) != 2:
        print(usage_message)
        sys.exit(1)

    if sys.argv[1] == "train":
        train_swingup()
    elif sys.argv[1] == "test":
        test_trained_swingup_policy()
    else:
        print(usage_message)
        sys.exit(1)
