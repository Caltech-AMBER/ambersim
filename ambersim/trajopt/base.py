from typing import Tuple

import jax
from flax import struct
from jax import grad, hessian

# ####################### #
# TRAJECTORY OPTIMIZATION #
# ####################### #


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
    Similar logic applies for not specifying the role of the CostFunction - we trust that the user will either use the
    provided API or will ignore it and still end up implementing something custom and reasonable.

    Finally, abstract dataclasses are weird, so we just make all children implement the below functions by instead
    raising a NotImplementedError.
    """

    def optimize(self, params: TrajectoryOptimizerParams) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Optimizes a trajectory.

        The shapes of the outputs include (?) because we may choose to return non-zero-order-hold parameterizations of
        the optimized trajectories (for example, we could choose to return a cubic spline parameterization of the
        control inputs over the trajectory as is done in the gradient-based methods of MJPC).

        Args:
            params: The parameters of the trajectory optimizer.

        Returns:
            xs_star (shape=(N + 1, nq + nv) or (?)): The optimized trajectory.
            us_star (shape=(N, nu) or (?)): The optimized controls.
        """
        raise NotImplementedError


# ############# #
# COST FUNCTION #
# ############# #


@struct.dataclass
class CostFunctionParams:
    """Generic parameters for cost functions."""


@struct.dataclass
class CostFunction:
    """The API for generic cost functions for trajectory optimization problems for mechanical systems.

    Rationale behind CostFunctionParams in this generic API:
    (1) computation of higher-order derivatives could depend on results or intermediates from lower-order derivatives.
        So, we can flexibly cache the requisite values to avoid repeated computation;
    (2) we may want to randomize or optimize the cost function parameters themselves, so specifying a generic pytree
        as input generically accounts for all possibilities;
    (3) there could simply be parameters that cannot be easily specified in advance that are key for cost evaluation,
        like a time-varying reference trajectory that gets updated in real time.
    (4) histories of higher-order derivatives can be useful for updating their current estimates, e.g., BFGS.
    """

    def cost(self, xs: jax.Array, us: jax.Array, params: CostFunctionParams) -> Tuple[jax.Array, CostFunctionParams]:
        """Computes the cost of a trajectory.

        Args:
            xs (shape=(N + 1, nq + nv)): The state trajectory.
            us (shape=(N, nu)): The controls over the trajectory.
            params: The parameters of the cost function.

        Returns:
            val (shape=(,)): The cost of the trajectory.
            new_params: The updated parameters of the cost function.
        """
        raise NotImplementedError

    def grad(
        self, xs: jax.Array, us: jax.Array, params: CostFunctionParams
    ) -> Tuple[jax.Array, jax.Array, CostFunctionParams, CostFunctionParams]:
        """Computes the gradient of the cost of a trajectory.

        The default implementation of this function uses JAX's autodiff. Simply override this function if you would like
        to supply an analytical gradient.

        Args:
            xs (shape=(N + 1, nq + nv)): The state trajectory.
            us (shape=(N, nu)): The controls over the trajectory.
            params: The parameters of the cost function.

        Returns:
            gcost_xs (shape=(N + 1, nq + nv): The gradient of the cost wrt xs.
            gcost_us (shape=(N, nu)): The gradient of the cost wrt us.
            gcost_params: The gradient of the cost wrt params.
            new_params: The updated parameters of the cost function.
        """
        _fn = lambda xs, us, params: self.cost(xs, us, params)[0]  # only differentiate wrt the cost val
        return grad(_fn, argnums=(0, 1, 2))(xs, us, params) + (params,)

    def hess(
        self, xs: jax.Array, us: jax.Array, params: CostFunctionParams
    ) -> Tuple[
        jax.Array, jax.Array, CostFunctionParams, jax.Array, CostFunctionParams, CostFunctionParams, CostFunctionParams
    ]:
        """Computes the Hessian of the cost of a trajectory.

        The default implementation of this function uses JAX's autodiff. Simply override this function if you would like
        to supply an analytical Hessian.

        Let t, s be times 0, 1, 2, etc. Then, d^2H/da_{t,i}db_{s,j} = Hcost_asbs[t, i, s, j].

        Args:
            xs (shape=(N + 1, nq + nv)): The state trajectory.
            us (shape=(N, nu)): The controls over the trajectory.
            params: The parameters of the cost function.

        Returns:
            Hcost_xsxs (shape=(N + 1, nq + nv, N + 1, nq + nv)): The Hessian of the cost wrt xs.
            Hcost_xsus (shape=(N + 1, nq + nv, N, nu)): The Hessian of the cost wrt xs and us.
            Hcost_xsparams: The Hessian of the cost wrt xs and params.
            Hcost_usus (shape=(N, nu, N, nu)): The Hessian of the cost wrt us.
            Hcost_usparams: The Hessian of the cost wrt us and params.
            Hcost_paramsall: The Hessian of the cost wrt params and everything else.
            new_params: The updated parameters of the cost function.
        """
        _fn = lambda xs, us, params: self.cost(xs, us, params)[0]  # only differentiate wrt the cost val
        hessians = hessian(_fn, argnums=(0, 1, 2))(xs, us, params)
        Hcost_xsxs, Hcost_xsus, Hcost_xsparams = hessians[0]
        _, Hcost_usus, Hcost_usparams = hessians[1]
        Hcost_paramsall = hessians[2]
        return Hcost_xsxs, Hcost_xsus, Hcost_xsparams, Hcost_usus, Hcost_usparams, Hcost_paramsall, params
