import pickle
from pathlib import Path

import jax
import pytest
from brax.training import distribution
from brax.training.agents.ppo.networks import PPONetworks

from ambersim.learning.architectures import MLP
from ambersim.rl.helpers import BraxPPONetworksWrapper
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


def test_ppo_networks_io():
    """Test saving and loading PPONetworks."""
    # Create a PPONetworks object
    observation_size = 3
    action_size = 2
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

    # Save to a file
    local_dir = Path("_test_ppo_networks_io")
    local_dir.mkdir(parents=True, exist_ok=True)
    model_path = local_dir / "ppo_networks.pkl"
    with Path(model_path).open("wb") as f:
        pickle.dump(network_wrapper, f)

    # Load from a file and check that the network is the same
    with Path(model_path).open("rb") as f:
        new_network_wrapper = pickle.load(f)
    new_ppo_networks = new_network_wrapper.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
    )
    assert isinstance(new_ppo_networks, PPONetworks)
    assert jax.tree_util.tree_structure(ppo_networks) == jax.tree_util.tree_structure(new_ppo_networks)

    _rmtree(local_dir)
