import functools

import jax.numpy as jp
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train
from flax import struct


@struct.dataclass
class PPOConfig:
    """Config for ppo network param."""

    # Configuration for PPO Networks
    policy_hidden_layer_sizes: tuple = (64, 64, 64)

    # Configuration for PPO Training
    num_timesteps: int = 10000
    num_evals: int = 10
    reward_scaling: float = 1.0
    episode_length: int = 500
    normalize_observations: bool = True
    action_repeat: int = 1
    unroll_length: int = 20
    num_minibatches: int = 8
    gae_lambda: float = 0.95
    num_updates_per_batch: int = 5
    discounting: float = 0.96
    learning_rate: float = 1e-4
    entropy_cost: float = 1e-2
    num_envs: int = 512
    batch_size: int = 512
    num_resets_per_eval: int = 5
    seed: int = 0
    # Add any additional fields here


def make_networks_factory(config: PPOConfig):
    """Wrapper for constructing ppo network."""
    return functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=config.policy_hidden_layer_sizes)


def train_fn(config: PPOConfig):
    """Wrapper for constructing training function for ppo."""
    return functools.partial(
        ppo_train,
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
