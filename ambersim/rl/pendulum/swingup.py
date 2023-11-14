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
    # in pendulum.xml, we use a time step of 0.001, which is the "framerate of reality"
    # divide 1khz by physics_steps_per_control_step to get the control frequency
    physics_steps_per_control_step: int = 10

    # the standard deviation of the noise (in radians) to add to the angular observations
    stdev_obs: float = 0.0

    # the angular tolerance for the task to be considered "done" (in radians)
    tol_done: float = 1e-2

    # max episode length
    T_reset: int = 1000


class PendulumSwingupEnv(MjxEnv):
    """Environment for training a torque-constrained pendulum swingup task.

    This is the most dead simple swingup task: simply take a pendulum starting from hanging and try to go vertical.

    States: x = (qpos, qvel), shape=(2,)
    Observations: y = (cos(theta), sin(theta), dtheta), shape=(3,)
    Actions: a = tau, the motor torque, shape=(1,)
    """

    def __init__(self, config: Optional[PendulumSwingupConfig] = None) -> None:
        """Initialize the swingup env. See parent docstring."""
        if config is None:
            config = PendulumSwingupConfig()
        self.config = config
        mj_model = load_mj_model_from_file(config.model_path)
        self._init_q = mj_model.keyframe("home").qpos

        super().__init__(
            mj_model,
            config.physics_steps_per_control_step,
        )

    def _theta_domain(self, theta: jax.Array) -> jax.Array:
        """Restricts the value of an angle to [0, 2 * pi]."""
        return jnp.mod(theta + jnp.pi, 2 * jnp.pi) - jnp.pi

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
        dtheta = data.qvel[0]
        u = data.ctrl[0]
        return -(self._theta_domain(theta) ** 2 + 0.1 * dtheta**2 + 0.001 * u**2)

    def reset(self, rng: jax.Array) -> State:
        """Resets the env. See parent docstring."""
        rng, key = jax.random.split(rng)

        # resetting the positions and velocities
        qpos = jnp.array(self._init_q)
        qvel = jnp.zeros(self.model.nv)
        data = self.pipeline_init(qpos, qvel)

        # other State fields
        obs = self.compute_obs(data, {})
        reward, done = jnp.zeros(2)
        metrics = {}
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
        done = jnp.where(jnp.abs(self._theta_domain(data.qpos[0])) < self.config.tol_done, 1.0, 0.0)

        # updating state
        state.info["step"] = jnp.where(
            (done == 1.0) | (state.info["step"] > self.config.T_reset), 0, state.info["step"] + 1
        )  # reset step counter if done
        state.info.update(rng=rng)
        state = state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)
        return state
