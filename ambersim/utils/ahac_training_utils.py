import functools

import jax.numpy as jp
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax import struct
from ambersim.utils.ahac_train_1 import *
from ambersim.utils.ahac import AHAC

@struct.dataclass
class PPOConfig:
    """Config for ppo network param."""

    # Configuration for PPO Networks
    policy_hidden_layer_sizes: tuple = (64, 64, 64)

    # Configuration for PPO Training        
#    num_timesteps: int = 10_000_000
#    num_evals: int = 40
#    num_timesteps: int = 100_000_000
#    num_evals: int = 160
    num_timesteps: int = 100_000_000
    num_evals: int = 400
    reward_scaling: float = 1.0
    episode_length: int = 128
##    normalize_observations: bool = True
    normalize_observations: bool = False
    action_repeat: int = 1  # Assuming no change needed
##    unroll_length: int = 32  # Assuming no change needed
    unroll_length: int = 128 # Assuming no change needed
    num_minibatches: int = 16
    gae_lambda: float = 0.98
    num_updates_per_batch: int = 16  # Assuming no change needed
    discounting: float = 0.98  # Assuming no change needed
    learning_rate: float = 8e-4
    entropy_cost: float = 2e-3
##    num_envs: int = 4096  # Assuming no change needed
    num_envs: int = 256  # Assuming no change needed
    batch_size: int = 16
##    num_resets_per_eval: int = 5  # Assuming no change needed
    num_resets_per_eval: int = 1  # Assuming no change needed
    seed: int = 0  # Assuming no change needed
    # Add any additional fields here


def make_networks_factory(config: PPOConfig):
    """Wrapper for constructing ppo network."""
    return functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=config.policy_hidden_layer_sizes)


def train_fn(config: PPOConfig, env=None):
    """Wrapper for constructing training function for ppo."""
    # TODO: replace with new training function
    ahac = AHAC(env=env,
                actor_config={},
                critic_config={},
                steps_min=20,  # minimum horizon
                steps_max=50,  # maximum horizon
                max_epochs=10,  # number of short rollouts to do (i.e. epochs)
                train=True,  # if False, we only eval the policy
                logdir="./ahac_logs",)
    # ahac.train()  
    # import ipdb; ipdb.set_trace()
    # train(
    #     num_timesteps=config.num_timesteps,
    #     episode_length=config.episode_length,
    #     environment=env)

    return ahac.train


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
