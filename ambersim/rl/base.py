from abc import ABC, abstractmethod
from typing import Any, Dict

import jax
import mujoco
import numpy as np
from brax.base import Base, Motion, Transform
from brax.envs.base import Env
from flax import struct
from jax import numpy as jp
from mujoco import mjx


@struct.dataclass
class State(Base):
    """Environment state for training and inference with brax.

    Args:
        pipeline_state: the physics state.
        obs: environment observations.
        reward: environment reward.
        done: True if the current episode has terminated.
        metrics: metrics that get tracked per environment step.
        info: environment variables defined and updated by the environment reset and step functions.
    """

    pipeline_state: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


class MjxEnv(Env, ABC):
    """API for an MJX system for training and inference in brax."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        physics_steps_per_control_step: int = 1,
    ) -> None:
        """Initializes MjxEnv.

        Args:
            mj_model: The MuJoCo model.
            physics_steps_per_control_step: The number of times to step the physics pipeline for each environment step.
        """
        assert physics_steps_per_control_step >= 1
        self.model = mj_model
        self.data = mujoco.MjData(mj_model)
        self.sys = mjx.device_put(mj_model)
        self._physics_steps_per_control_step = physics_steps_per_control_step

    @property
    def dt(self) -> jax.Array:
        """The timestep used for each env step."""
        return self.sys.opt.timestep * self._physics_steps_per_control_step

    @property
    def observation_size(self) -> int:
        """Dimension of the observation space."""
        rng = jax.random.PRNGKey(0)
        reset_state = self.unwrapped.reset(rng)
        return reset_state.obs.shape[-1]

    @property
    def action_size(self) -> int:
        """Dimension of the action space."""
        return self.sys.nu

    @property
    def backend(self) -> str:
        """The brax physics backend.

        The options are ['generalized', 'spring', 'positional', 'mjx'].
        See: github.com/google/brax/tree/16304037a36b1d9c8c0b3084f57d1159627b636b#one-api-three-pipelines
        """
        return "mjx"

    def pipeline_init(self, qpos: jax.Array, qvel: jax.Array) -> mjx.Data:
        """Initializes the physics state."""
        data = mjx.device_put(self.data)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jp.zeros(self.sys.nu))
        data = mjx.forward(self.sys, data)
        return data

    def pipeline_step(self, data: mjx.Data, ctrl: jax.Array) -> mjx.Data:
        """Takes a physics step using the physics pipeline."""

        def f(data, _):
            data = data.replace(ctrl=ctrl)
            return mjx.step(self.sys, data), None

        data, _ = jax.lax.scan(f, data, (), self._physics_steps_per_control_step)
        return data

    def compute_reward(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Computes the reward for the current environment state. May modify state in place.

        Args:
            data: The physics state.
            info: Auxiliary info from the State.

        Returns:
            reward: The reward.

        Raises:
            NotImplementedError: If the environment does not implement a reward function.
            We choose to not make this abstract in case classes that inherit from MjxEnv are used for simulation but not
            for reinforcement learning.
        """
        raise NotImplementedError

    def compute_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        """Observes the environment based on the system State. May modify state in place.

        Args:
            data: The physics state.
            info: Auxiliary info from the State.

        Returns:
            obs: the observation.

        Raises:
            NotImplementedError: if the environment does not implement an observation function.
            We choose to not make this abstract in case classes that inherit from MjxEnv are used for simulation but not
            for reinforcement learning.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment.

        Args:
            rng: random number generator seed.

        Returns:
            state: initial environment state.
        """

    @abstractmethod
    def step(self, state: State, action: jax.Array) -> State:
        """Takes a step in the environment.

        Args:
            state: current environment state.
            action: action to take.

        Returns:
            new_state: new environment state.
        """
