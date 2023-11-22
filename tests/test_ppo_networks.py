import jax
import jax.numpy as jnp
import pytest
from brax.training import distribution
from brax.training.agents.ppo.networks import PPONetworks

from ambersim.rl.networks import MLP, BraxPPONetworksWrapper
from ambersim.utils._internal_utils import _rmtree


def test_ppo_wrapper():
    """Test the BraxPPONetworksWrapper."""
    observation_size = 3
    action_size = 2

    with pytest.raises(AssertionError):
        # We should get an error if the policy network's output doesn't match
        # the size of the action distribution (mean + variance)
        network_wrapper = BraxPPONetworksWrapper(
            policy_network=MLP(layer_sizes=(512, 2)),
            value_network=MLP(layer_sizes=(512, 1)),
            action_distribution=distribution.NormalTanhDistribution,
        )
        network_wrapper.make_ppo_networks(
            observation_size=observation_size,
            action_size=action_size,
        )

    with pytest.raises(AssertionError):
        # We should get an error if the value network's output isn't 1D
        network_wrapper = BraxPPONetworksWrapper(
            policy_network=MLP(layer_sizes=(512, 4)),
            value_network=MLP(layer_sizes=(512, 2)),
            action_distribution=distribution.NormalTanhDistribution,
        )
        network_wrapper.make_ppo_networks(
            observation_size=observation_size,
            action_size=action_size,
        )

    # We should end up with a PPONetworks object if everything is correct
    network_wrapper = BraxPPONetworksWrapper(
        policy_network=MLP(layer_sizes=(512, 4)),
        value_network=MLP(layer_sizes=(512, 1)),
        action_distribution=distribution.NormalTanhDistribution,
    )
    ppo_networks = network_wrapper.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
    )
    assert isinstance(ppo_networks, PPONetworks)
