import functools

import jax.numpy as jp
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax import struct


@struct.dataclass
class PPOConfig:
    """Config for ppo network param."""

    # Configuration for PPO Networks
    policy_hidden_layer_sizes: tuple = (64, 64, 64)

    # Configuration for PPO Training
    num_timesteps: int = 1_000_000
    num_evals: int = 5
    reward_scaling: float = 1.0
    episode_length: int = 500
    normalize_observations: bool = True
    action_repeat: int = 1  # Assuming no change needed
    unroll_length: int = 32  # Assuming no change needed
    num_minibatches: int = 32
    gae_lambda: float = 0.98
    num_updates_per_batch: int = 10  # Assuming no change needed
    discounting: float = 0.98  # Assuming no change needed
    learning_rate: float = 9e-5
    entropy_cost: float = 2e-3
    num_envs: int = 4096  # Assuming no change needed
    batch_size: int = 512
    num_resets_per_eval: int = 5  # Assuming no change needed
    seed: int = 0  # Assuming no change needed
    # Add any additional fields here


def make_networks_factory(config: PPOConfig):
    """Wrapper for constructing ppo network."""
    return functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=config.policy_hidden_layer_sizes)


def train_fn(config: PPOConfig):
    """Wrapper for constructing training function for ppo."""
    return functools.partial(
        ppo.train,
        num_timesteps=config.num_timesteps,
        num_evals=config.num_evals,
        reward_scaling=config.reward_scaling,
        episode_length=config.episode_length,
        normalize_observations=config.normalize_observations,
        action_repeat=config.action_repeat,
        unroll_length=config.unroll_length,
        num_minibatches=config.num_minibatches,
        gae_lambda=config.gae_lambda,
        num_updates_per_batch=config.num_updates_per_batch,
        discounting=config.discounting,
        learning_rate=config.learning_rate,
        entropy_cost=config.entropy_cost,
        num_envs=config.num_envs,
        batch_size=config.batch_size,
        network_factory=make_networks_factory(config),
        num_resets_per_eval=config.num_resets_per_eval,
        seed=config.seed,
    )


def load_model(environment, network_factory, model_path):
    """Load the model parameters without training.

    Args:
      environment: The environment to use for the network.
      network_factory: Factory function to create networks.
      model_path: Path to the saved model parameters.

    Returns:
      A tuple containing the inference function and loaded parameters.
    """
    # Create the network (or networks) for the environment

    normalize = running_statistics.normalize

    ppo_network = network_factory(
        environment.config.history_size * environment.observation_size_single_step,
        environment.action_size,
        preprocess_observations_fn=normalize,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    # Load the parameters
    params = model.load_params(model_path)

    return make_policy, params
