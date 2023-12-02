from typing import Tuple

import jax
from flax import struct


@struct.dataclass
class TrajectoryOptimizerParams:
    """The parameters for generic trajectory optimization algorithms.

    Parameters we may want to optimize should be included here.

    This is left completely empty to allow for maximum flexibility in the API. Some examples:
            - A direct collocation method might have parameters for the number of collocation points, the collocation
              scheme, and the number of optimization iterations.
            - A shooting method might have parameters for the number of shooting points, the shooting scheme, and the number
              of optimization iterations.
    The parameters also include initial iterates for each type of algorithm. Some examples:
            - A direct collocation method might have initial iterates for the controls and the state trajectory.
            - A shooting method might have initial iterates for the controls only.

    Parameters which we want to remain untouched by JAX transformations can be marked by pytree_node=False, e.g.,
            ```
            @struct.dataclass
            class ChildParams:
                    ...
                    # example field
                    example: int = struct.field(pytree_node=False)
                    ...
            ```
    """


@struct.dataclass
class TrajectoryOptimizer:
    """The API for generic trajectory optimization algorithms."""

    def __init__(self) -> None:
        """Initialize the trajopt object."""

    @staticmethod
    def optimize(params: TrajectoryOptimizerParams) -> Tuple[jax.Array, jax.Array]:
        """Optimizes a trajectory.

        Args:
                params: The parameters of the trajectory optimizer.

        Returns:
                xs (shape=(N + 1, nq + nv)): The optimized trajectory.
                us (shape=(N, nu)): The optimized controls.
        """
        # abstract dataclasses are weird, so we just make all children implement this - to be useful, they need it
        # anyway, so it isn't really a problem if an "abstract" TrajectoryOptimizer is instantiated.
        raise NotImplementedError
