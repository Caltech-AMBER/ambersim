from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from mujoco import mjx

from ambersim.control.base import Controller, ControllerParams
from ambersim.trajopt.base import TrajectoryOptimizer
from ambersim.trajopt.shooting import PDPredictiveSamplerParams, PredictiveSampler, VanillaPredictiveSamplerParams

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
class VanillaPredictiveSamplingControllerParams(PredictiveControllerParams):
    """Vanilla predictive sampling controller params."""

    key: jax.Array  # random key for sampling
    x: jax.Array  # shape=(nq + nv,) current state
    guess: jax.Array  # shape=(N, nu) current guess


@struct.dataclass
class VanillaPredictiveSamplingController(PredictiveController):
    """Vanilla predictive sampling controller."""

    def __post_init__(self) -> None:
        """Post-initialization check."""
        assert isinstance(
            self.trajectory_optimizer, PredictiveSampler
        ), "trajectory_optimizer must be a PredictiveSampler!"

    def compute(self, ctrl_params: VanillaPredictiveSamplingControllerParams) -> jax.Array:
        """Computes a control input using forward prediction.

        Args:
            ctrl_params: Inputs into the controller.

        Returns:
            u (shape=(nu,)): The control input.
        """
        return self.compute_with_us_star(ctrl_params)[0]

    def compute_with_us_star(
        self, ctrl_params: VanillaPredictiveSamplingControllerParams
    ) -> Tuple[jax.Array, jax.Array]:
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
            guess=ctrl_params.guess,
        )
        xs_star, us_star = self.trajectory_optimizer.optimize(to_params)
        u = us_star[0, :]
        return u, us_star


@struct.dataclass
class PDPredictiveSamplingControllerParams(PredictiveControllerParams):
    """PD predictive sampling controller params."""

    key: jax.Array  # random key for sampling
    x: jax.Array  # shape=(nq + nv,) current state
    guess: jax.Array  # shape=(N, nq) current guess
    kp: float  # proportional gain
    kd: float  # derivative gain


@struct.dataclass
class PDPredictiveSamplingController(PredictiveController):
    """PD predictive sampling controller."""

    def __post_init__(self) -> None:
        """Post-initialization check."""
        assert isinstance(
            self.trajectory_optimizer, PredictiveSampler
        ), "trajectory_optimizer must be a PredictiveSampler!"

    def compute(self, ctrl_params: PDPredictiveSamplingControllerParams) -> jax.Array:
        """Computes a control input using forward prediction.

        Args:
            ctrl_params: Inputs into the controller.

        Returns:
            u (shape=(nu,)): The control input.
        """
        return self.compute_with_us_star(ctrl_params)[0]

    def compute_with_us_star(self, ctrl_params: PDPredictiveSamplingControllerParams) -> Tuple[jax.Array, jax.Array]:
        """Computes a control input using forward prediction + the optimal sequence of guesses.

        This is needed in practice because the current optimal sequence is used to warm start the sampling distribution
        for the next call of the controller.

        Args:
            ctrl_params: Inputs into the controller.

        Returns:
            u (shape=(nu,)): The control input.
            us_star (shape=(N, nu)): The optimal control sequence.
        """
        to_params = PDPredictiveSamplerParams(
            key=ctrl_params.key,
            x0=ctrl_params.x,
            guess=ctrl_params.guess,
            kp=ctrl_params.kp,
            kd=ctrl_params.kd,
        )
        xs_star, us_star = self.trajectory_optimizer.optimize(to_params)
        u = us_star[0, :]
        return u, us_star
