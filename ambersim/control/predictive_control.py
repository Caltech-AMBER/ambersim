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
class PredictiveSamplingControllerParams(ControllerParams):
    """Predictive sampling controller params."""

    key: jax.Array  # random key for sampling
    x: jax.Array  # shape=(nq + nv,) current state
    N: int  # prediction horizon

    def __post_init__(self) -> None:
        """Post-initialization check."""
        assert isinstance(
            self.trajectory_optimizer, VanillaPredictiveSampler
        ), "trajectory_optimizer must be a VanillaPredictiveSampler!"


@struct.dataclass
class PredictiveSamplingController(Controller):
    """Predictive sampling controller."""

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
            us_guess=jnp.zeros((ctrl_params.N, self.model.nu)),
        )
        _, us_star = self.trajectory_optimizer.optimize(to_params)
        u = us_star[0, :]  # (nu,)
        return u
