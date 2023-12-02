from typing import Tuple

import jax
from flax import struct
from mujoco import mjx

from ambersim.trajopt.base import TrajectoryOptimizer, TrajectoryOptimizerParams


@struct.dataclass
class ShootingParams(TrajectoryOptimizerParams):
    """Parameters for shooting methods."""

    model = mjx.Model


@struct.dataclass
class ShootingAlgorithm(TrajectoryOptimizer):
    """A trajectory optimization algorithm based on shooting methods."""

    @staticmethod
    def optimize(params: ShootingParams) -> Tuple[jax.Array, jax.Array]:
        """Optimizes a trajectory.

        Args:
            params: The parameters of the trajectory optimizer.

        Returns:
            xs (shape=(N + 1, nq + nv)): The optimized trajectory.
            us (shape=(N, nu)): The optimized controls.
        """
