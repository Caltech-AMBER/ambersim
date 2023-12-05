from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import lax, vmap

from ambersim.trajopt.base import CostFunction, CostFunctionParams

"""A collection of common cost functions."""


class StaticGoalQuadraticCost(CostFunction):
    """A quadratic cost function that penalizes the distance to a static goal.

    This is the most vanilla possible quadratic cost. The cost matrices are static (defined at init time) and so is the
    single, fixed goal. The gradient is as compressed as it can be in general (one matrix multiplication), but the
    Hessian can be far more compressed by simplying referencing Q, Qf, and R - this implementation is inefficient and
    dense.
    """

    def __init__(self, Q: jax.Array, Qf: jax.Array, R: jax.Array, xg: jax.Array) -> None:
        """Initializes a quadratic cost function.

        Args:
            Q (shape=(nx, nx)): The state cost matrix.
            Qf (shape=(nx, nx)): The final state cost matrix.
            R (shape=(nu, nu)): The control cost matrix.
            xg (shape=(nq,)): The goal state.
        """
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.xg = xg

    @staticmethod
    def batch_quadform(bs: jax.Array, A: jax.Array) -> jax.Array:
        """Computes a batched quadratic form for a single instance of A.

        Args:
            bs (shape=(..., n)): The batch of vectors.
            A (shape=(n, n)): The matrix.

        Returns:
            val (shape=(...,)): The batch of quadratic forms.
        """
        return jnp.einsum("...i,ij,...j->...", bs, A, bs)

    @staticmethod
    def batch_matmul(bs: jax.Array, A: jax.Array) -> jax.Array:
        """Computes a batched matrix multiplication for a single instance of A.

        Args:
            bs (shape=(..., n)): The batch of vectors.
            A (shape=(n, n)): The matrix.

        Returns:
            val (shape=(..., n)): The batch of matrix multiplications.
        """
        return jnp.einsum("...i,ij->...j", bs, A)

    def cost(self, xs: jax.Array, us: jax.Array, params: CostFunctionParams) -> Tuple[jax.Array, CostFunctionParams]:
        """Computes the cost of a trajectory.

        cost = 0.5 * (xs - xg)' @ Q @ (xs - xg) + 0.5 * us' @ R @ us

        Args:
            xs (shape=(N + 1, nq + nv)): The state trajectory.
            us (shape=(N, nu)): The controls over the trajectory.
            params: Unused. Included for API compliance.

        Returns:
            cost_val: The cost of the trajectory.
            new_params: Unused. Included for API compliance.
        """
        xs_err = xs[:-1, :] - self.xg  # errors before the terminal state
        xf_err = xs[-1, :] - self.xg
        val = 0.5 * jnp.squeeze(
            (
                jnp.sum(self.batch_quadform(xs_err, self.Q))
                + self.batch_quadform(xf_err, self.Qf)
                + jnp.sum(self.batch_quadform(us, self.R))
            )
        )
        return val, params

    def grad(
        self, xs: jax.Array, us: jax.Array, params: CostFunctionParams
    ) -> Tuple[jax.Array, jax.Array, CostFunctionParams, CostFunctionParams]:
        """Computes the gradient of the cost of a trajectory.

        Args:
            xs (shape=(N + 1, nq + nv)): The state trajectory.
            us (shape=(N, nu)): The controls over the trajectory.
            params: Unused. Included for API compliance.

        Returns:
            gcost_xs (shape=(N + 1, nq + nv): The gradient of the cost wrt xs.
            gcost_us (shape=(N, nu)): The gradient of the cost wrt us.
            gcost_params: Unused. Included for API compliance.
            new_params: Unused. Included for API compliance.
        """
        xs_err = xs[:-1, :] - self.xg  # errors before the terminal state
        xf_err = xs[-1, :] - self.xg
        gcost_xs = jnp.concatenate(
            (
                self.batch_matmul(xs_err, self.Q),
                (self.Qf @ xf_err)[None, :],
            ),
            axis=-2,
        )
        gcost_us = self.batch_matmul(us, self.R)
        return gcost_xs, gcost_us, params, params

    def hess(
        self, xs: jax.Array, us: jax.Array, params: CostFunctionParams
    ) -> Tuple[
        jax.Array, jax.Array, CostFunctionParams, jax.Array, CostFunctionParams, CostFunctionParams, CostFunctionParams
    ]:
        """Computes the gradient of the cost of a trajectory.

        Let t, s be times 0, 1, 2, etc. Then, d^2H/da_{t,i}db_{s,j} = Hcost_asbs[t, i, s, j].

        Args:
            xs (shape=(N + 1, nq + nv)): The state trajectory.
            us (shape=(N, nu)): The controls over the trajectory.
            params: Unused. Included for API compliance.

        Returns:
            Hcost_xsxs (shape=(N + 1, nq + nv, N + 1, nq + nv)): The Hessian of the cost wrt xs.
            Hcost_xsus (shape=(N + 1, nq + nv, N, nu)): The Hessian of the cost wrt xs and us.
            Hcost_xsparams: The Hessian of the cost wrt xs and params.
            Hcost_usus (shape=(N, nu, N, nu)): The Hessian of the cost wrt us.
            Hcost_usparams: The Hessian of the cost wrt us and params.
            Hcost_paramsall: The Hessian of the cost wrt params and everything else.
            new_params: The updated parameters of the cost function.
        """
        # setting up
        nx = self.Q.shape[0]
        N, nu = us.shape
        Q = self.Q
        Qf = self.Qf
        R = self.R
        dummy_params = CostFunctionParams()

        # Hessian for state
        Hcost_xsxs = jnp.zeros((N + 1, nx, N + 1, nx))
        Hcost_xsxs = vmap(
            lambda i: lax.dynamic_update_slice(
                jnp.zeros((nx, N + 1, nx)),
                Q[:, None, :],
                (0, i, 0),
            )
        )(
            jnp.arange(N + 1)
        )  # only the terms [i, :, i, :] are nonzero
        Hcost_xsxs = Hcost_xsxs.at[-1, :, -1, :].set(Qf)  # last one is different

        # trivial cross-terms of Hessian
        Hcost_xsus = jnp.zeros((N + 1, nx, N, nu))
        Hcost_xsparams = dummy_params

        # Hessian for control inputs
        Hcost_usus = jnp.zeros((N, nu, N, nu))
        Hcost_usus = vmap(
            lambda i: lax.dynamic_update_slice(
                jnp.zeros((nu, N, nu)),
                R[:, None, :],
                (0, i, 0),
            )
        )(
            jnp.arange(N)
        )  # only the terms [i, :, i, :] are nonzero

        # trivial cross-terms and Hessian for params
        Hcost_usparams = dummy_params
        Hcost_paramsall = dummy_params
        return Hcost_xsxs, Hcost_xsus, Hcost_xsparams, Hcost_usus, Hcost_usparams, Hcost_paramsall, params
