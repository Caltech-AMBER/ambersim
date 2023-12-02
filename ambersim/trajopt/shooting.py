import jax
from flax import struct

from ambersim.trajopt.base import TrajectoryOptimizer, TrajectoryOptimizerParams


@struct.dataclass
class ShootingParams(TrajectoryOptimizerParams):
    """Parameters for shooting methods."""


@struct.dataclass
class ShootingAlgorithm(TrajectoryOptimizer):
    """A trajectory optimization algorithm based on shooting methods."""
