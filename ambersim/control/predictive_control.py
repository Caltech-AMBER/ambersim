from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from mujoco import mjx

from ambersim.control.base import Controller, ControllerParams
from ambersim.trajopt.base import TrajectoryOptimizer
from ambersim.trajopt.shooting import VanillaPredictiveSampler, VanillaPredictiveSamplerParams

# ########### #
# GENERIC API #
# ########### #


@struct.dataclass
class PredictiveControllerParams(ControllerParams):
    """The generic API for predictive controller params."""


@struct.dataclass
class PredictiveController(Controller):
    """The generic API for a predictive controller."""

    trajectory_optimizer: TrajectoryOptimizer
    model: mjx.Model

    def compute(self, ctrl_params: PredictiveControllerParams) -> jax.Array:
        """Computes a control input using forward prediction."""
        raise NotImplementedError


# ################### #
# PREDICTIVE SAMPLING #
# ################### #


@struct.dataclass
class PredictiveSamplingControllerParams(PredictiveControllerParams):
    """Predictive sampling controller params."""

    key: jax.Array  # random key for sampling
    x: jax.Array  # shape=(nq + nv,) current state
    us_guess: jax.Array  # shape=(N, nu) current guess


@struct.dataclass
class PredictiveSamplingController(PredictiveController):
    """Predictive sampling controller."""

    def __post_init__(self) -> None:
        """Post-initialization check."""
        assert isinstance(
            self.trajectory_optimizer, VanillaPredictiveSampler
        ), "trajectory_optimizer must be a VanillaPredictiveSampler!"  # TODO(ahl): this is too restrictive

    def compute(self, ctrl_params: PredictiveSamplingControllerParams) -> jax.Array:
        """Computes a control input using forward prediction.

        Args:
            ctrl_params: Inputs into the controller.

        Returns:
            u (shape=(nu,)): The control input.
        """
        to_params = VanillaPredictiveSamplerParams(
            key=ctrl_params.key,
            x0=ctrl_params.x,
            us_guess=ctrl_params.us_guess,
        )
        _, us_star = self.trajectory_optimizer.optimize(to_params)
        u = us_star[0, :]  # (nu,)
        return u

    def compute_with_us_star(self, ctrl_params: PredictiveSamplingControllerParams) -> Tuple[jax.Array, jax.Array]:
        """Computes a control input using forward prediction + the optimal sequence of guesses.

        This is needed in practice because the current optimal sequence is used to warm start the sampling distribution
        for the next call of the controller.

        Args:
            ctrl_params: Inputs into the controller.

        Returns:
            u (shape=(nu,)): The control input.
            us_star (shape=(N, nu)): The optimal control sequence.
        """
        to_params = VanillaPredictiveSamplerParams(
            key=ctrl_params.key,
            x0=ctrl_params.x,
            us_guess=ctrl_params.us_guess,
        )
        xs_star, us_star = self.trajectory_optimizer.optimize(to_params)
        u = us_star[0, :]
        return u, us_star
