from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
import mujoco as mj
from flax import struct
from mujoco import mjx

from ambersim.rl.base import MjxEnv, State
from ambersim.utils.io_utils import load_mj_model_from_file


@struct.dataclass
class PendulumSwingupConfig:
    """Config dataclass for pendulum swingup."""

    # model path: scene.xml contains ground + other niceties in addition to the pendulum
    model_path: Union[Path, str] = "models/pendulum/scene.xml"

    # number of "simulation steps" for every control input
    physics_steps_per_control_step: int = 1

    # the standard deviation of the noise (in radians) to add to the angular observations
    stdev_obs: float = 0.0

    # Reward function coefficients
    theta_cost_weight: float = 1.0
    theta_dot_cost_weight: float = 0.1
    control_cost_weight: float = 0.001

    # Ranges for sampling initial conditions
    qpos_hi: float = jnp.pi
    qpos_lo: float = -jnp.pi
    qvel_hi: float = 2
    qvel_lo: float = -2


class PendulumSwingupEnv(MjxEnv):
    """Environment for training a torque-constrained pendulum swingup task.

    This is the most dead simple swingup task: simply take a pendulum starting
    from hanging and try to go vertical.

    States: x = (theta, dtheta), shape=(2,)
    Observations: y = (cos(theta), sin(theta), dtheta), shape=(3,)
    Actions: a = tau, the motor torque, shape=(1,)
    """

    def __init__(self, config: Optional[PendulumSwingupConfig] = None) -> None:
        """Initialize the swingup env. See parent docstring."""
        if config is None:
            config = PendulumSwingupConfig()
        self.config = config
        mj_model = load_mj_model_from_file(config.model_path)

        super().__init__(
            mj_model,
            config.physics_steps_per_control_step,
        )

    def compute_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Observes the environment based on the system State. See parent docstring."""
        theta = data.qpos[0]
        obs = jnp.stack((jnp.cos(theta), jnp.sin(theta), data.qvel[0]))
        return obs

    def compute_reward(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Computes the reward for the current environment state.

        Returns:
            reward (shape=(1,)): the reward, maximized at qpos[0] = np.pi.
        """
        theta = data.qpos[0]
        theta_dot = data.qvel[0]
        tau = data.ctrl[0]

        # Compute a normalized theta error
        theta_err = theta - jnp.pi
        theta_err_normalized = jnp.arctan2(jnp.sin(theta_err), jnp.cos(theta_err))

        # Compute the reward
        reward_theta = -self.config.theta_cost_weight * jnp.square(theta_err_normalized).sum()
        reward_theta_dot = -self.config.theta_dot_cost_weight * jnp.square(theta_dot).sum()
        reward_tau = -self.config.control_cost_weight * jnp.square(tau).sum()

        return reward_theta + reward_theta_dot + reward_tau

    def reset(self, rng: jax.Array) -> State:
        """Resets the env. See parent docstring."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # reset the positions and velocities
        qpos = jax.random.uniform(rng1, (self.sys.nq,), minval=self.config.qpos_lo, maxval=self.config.qpos_hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=self.config.qvel_lo, maxval=self.config.qvel_hi)
        data = self.pipeline_init(qpos, qvel)

        # other state fields
        obs = self.compute_obs(data, {})
        reward, done = jnp.zeros(2)
        metrics = {"reward": reward}
        state_info = {"rng": rng, "step": 0}
        state = State(data, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Takes a step in the environment. See parent docstring."""
        rng, rng_obs = jax.random.split(state.info["rng"])

        # physics + observation + reward
        data = self.pipeline_step(state.pipeline_state, action)  # physics
        obs = self.compute_obs(data, state.info)  # observation
        obs = obs + jax.random.normal(rng_obs, obs.shape) * self.config.stdev_obs  # adding noise to obs
        reward = self.compute_reward(data, state.info)
        done = 0.0  # pendulum just runs for a fixed number of steps

        # updating state
        state.info["step"] = state.info["step"] + 1
        state.info["rng"] = rng
        state.metrics["reward"] = reward
        state = state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)
        return state
