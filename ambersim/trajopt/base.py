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

    Further, we choose not to specify the mjx.Model as either a field of this dataclass or as a parameter. The reason is
    because we want to allow for maximum flexibility in the API. Two motivating scenarios:
    (1) we want to domain randomize over the model parameters and potentially optimize for them. In this case, it makes
        sense to specify the mjx.Model as a parameter that gets passed as an input into the optimize function.
    (2) we want to fix the model and only randomize/optimize over non-model-parameters. For instance, this is the
        situation in vanilla predictive sampling. If we don't need to pass the model, we instead initialize it as a
        field of this dataclass, which makes the optimize function more performant, since it can just reference the
        fixed model attribute of the optimizer instead of applying JAX transformations to the entire large model pytree.

    Finally, abstract dataclasses are weird, so we just make all children implement the below functions by instead
    raising a NotImplementedError.
    """

    def cost(self, qs: jax.Array, vs: jax.Array, us: jax.Array) -> jax.Array:
        """Computes the cost of a trajectory.

        Args:
            qs: The generalized positions over the trajectory.
            vs: The generalized velocities over the trajectory.
            us: The controls over the trajectory.

        Returns:
            The cost of the trajectory.
        """
        raise NotImplementedError

    def optimize(self, params: TrajectoryOptimizerParams) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Optimizes a trajectory.

        The shapes of the outputs include (?) because we may choose to return non-zero-order-hold parameterizations of
        the optimized trajectories (for example, we could choose to return a cubic spline parameterization of the
        control inputs over the trajectory as is done in the gradient-based methods of MJPC).

        Args:
            params: The parameters of the trajectory optimizer.

        Returns:
            qs_star (shape=(N + 1, nq) or (?)): The optimized trajectory.
            vs_star (shape=(N + 1, nv) or (?)): The optimized generalized velocities.
            us_star (shape=(N, nu) or (?)): The optimized controls.
        """
        raise NotImplementedError
