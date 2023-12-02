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
    """The API for generic trajectory optimization algorithms on mechanical systems.

    We choose to implement this as a flax dataclass (as opposed to a regular class whose functions operate on pytree
    nodes) because:
    (1) the OOP formalism allows us to define coherent abstractions through inheritance;
    (2) struct.dataclass registers dataclasses a pytree nodes, so we can deal with awkward issues like the `self`
        variable when using JAX transformations on methods of the dataclass.
    """

    @staticmethod
    def optimize(params: TrajectoryOptimizerParams) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Optimizes a trajectory.

        The shapes of the outputs include (?) because we may choose to return non-zero-order-hold parameterizations of
        the optimized trajectories (for example, we could choose to return a cubic spline parameterization of the
        control inputs over the trajectory as is done in the gradient-based methods of MJPC).

        Args:
            params: The parameters of the trajectory optimizer.

        Returns:
            qs (shape=(N + 1, nq) or (?)): The optimized trajectory.
            vs (shape=(N + 1, nv) or (?)): The optimized generalized velocities.
            us (shape=(N, nu) or (?)): The optimized controls.
        """
        # abstract dataclasses are weird, so we just make all children implement this - to be useful, they need it
        # anyway, so it isn't really a problem if an "abstract" TrajectoryOptimizer is instantiated.
        raise NotImplementedError
