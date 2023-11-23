import flax.linen as nn
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.agents.ppo.networks import PPONetworks
from flax import struct


@struct.dataclass
class BraxPPONetworksWrapper:
    """A lightweight wrapper around brax's PPONetworks.

    Allows us to more easily save and load networks with non-default architectures.
    """

    policy_network: nn.Module
    value_network: nn.Module
    action_distribution: distribution.ParametricDistribution

    def make_ppo_networks(
        self,
        observation_size: int,
        action_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    ) -> PPONetworks:
        """Create a PPONetworks object, compatible with brax's ppo.train() function.

        Args:
            observation_size: Size of the input (observation).
            action_size: Size of the policy output (action).
            preprocess_observations_fn: Function to preprocess (e.g. normalize) observations.

        Returns:
            A PPONetworks object.
        """
        # Create an action distribution. The policy network should output the
        # parameters of this distribution.
        action_dist = self.action_distribution(event_size=action_size)

        # Set up a dummy observation and random key for size verifications
        dummy_observation = jnp.zeros((1, observation_size))
        rng = jax.random.PRNGKey(0)

        # Check that the output size of the policy network matches the size of
        # the action distribution.
        dummy_params = self.policy_network.init(rng, dummy_observation)
        policy_output = self.policy_network.apply(dummy_params, dummy_observation)
        assert (
            policy_output.shape[-1] == action_dist.param_size
        ), f"policy network output size {policy_output.shape[-1]} does not match action distribution size {action_dist.param_size}"

        # Create the policy network, a FeedForwardNetwork that contains an "init"
        # and an "apply" function.
        def policy_init(key):
            """Initialize the policy network from a random key."""
            return self.policy_network.init(key, dummy_observation)

        def policy_apply(processor_params, policy_params, obs):
            """Apply the policy given the parameters and an observation."""
            obs = preprocess_observations_fn(obs, processor_params)
            return self.policy_network.apply(policy_params, obs)

        # Create the value network. This is just like the policy network, but with a 1D output.
        dummy_value_params = self.value_network.init(rng, dummy_observation)
        value_output = self.value_network.apply(dummy_value_params, dummy_observation)
        assert (
            value_output.shape[-1] == 1
        ), f"value network output size {value_output.shape} does not match expected size 1"

        def value_init(key):
            """Initialize the value network from a random key."""
            return self.value_network.init(key, dummy_observation)

        def value_apply(processor_params, value_params, obs):
            """Apply the value function given the parameters and an observation."""
            obs = preprocess_observations_fn(obs, processor_params)
            return jnp.squeeze(self.value_network.apply(value_params, obs), axis=-1)

        return PPONetworks(
            policy_network=networks.FeedForwardNetwork(init=policy_init, apply=policy_apply),
            value_network=networks.FeedForwardNetwork(init=value_init, apply=value_apply),
            parametric_action_distribution=action_dist,
        )
