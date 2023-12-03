from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct

from ambersim.trajopt.base import CostFunction, CostFunctionParams

"""A collection of common cost functions."""


class StaticGoalQuadraticCost(CostFunction):
    """A quadratic cost function that penalizes the distance to a static goal.

    This is the most vanilla possible quadratic cost. The cost matrices are static (defined at init time) and so is the
    single, fixed goal. The gradient is as compressed as it can be in general (one matrix multiplication), but the
    Hessian can be far more compressed by simplying referencing Q, Qf, and R - this implementation is inefficient and
    dense.
    """

    def __init__(self, Q: jax.Array, Qf: jax.Array, R: jax.Array, qg: jax.Array, vg: jax.Array) -> None:
        """Initializes a quadratic cost function.

        Args:
            Q (shape=(nx, nx)): The state cost matrix.
            Qf (shape=(nx, nx)): The final state cost matrix.
            R (shape=(nu, nu)): The control cost matrix.
            qg (shape=(nq,)): The goal generalized coordinates.
            vg (shape=(nv,)): The goal generalized velocities.
        """
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.qg = qg
        self.vg = vg

    @staticmethod
    def _setup_util(
        qs: jax.Array, vs: jax.Array, qg: jax.Array, vg: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, int, int, int]:
        """Utility function that sets up the cost function.

        Args:
            qs (shape=(N + 1, nq)): The generalized positions over the trajectory.
            vs (shape=(N + 1, nv)): The generalized velocities over the trajectory.
            qg (shape=(nx,)): The goal generalized position.
            vg (shape=(nv,)): The goal generalized velocity.

        Returns:
            xs (shape=(N + 1, nx)): The states over the trajectory.
            xg (shape=(nx,)): The goal state.
            xs_err (shape=(N, nx)): The state errors up to the final state.
            xf_err (shape=(nx,)): The state error at the final state.
            nq: The number of generalized coordinates.
            nv: The number of generalized velocities.
        """
        xs = jnp.concatenate((qs, vs), axis=-1)
        xg = jnp.concatenate((qg, vg), axis=-1)
        xs_err = xs[:-1, :] - xg
        xf_err = xs[-1, :] - xg
        return xs, xg, xs_err, xf_err, qs.shape[-1], vs.shape[-1]

    @staticmethod
    def batch_quadform(bs: jax.Array, A: jax.array) -> jax.Array:
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

    def cost(
        self, qs: jax.Array, vs: jax.Array, us: jax.Array, params: CostFunctionParams
    ) -> Tuple[jax.Array, CostFunctionParams]:
        """Computes the cost of a trajectory.

        cost = 0.5 * ([q; v] - [qg; vg])' @ Q @ ([q; v] - [qg; vg]) + 0.5 * u' @ R @ u

        Args:
            qs (shape=(N + 1, nq)): The generalized positions over the trajectory.
            vs (shape=(N + 1, nv)): The generalized velocities over the trajectory.
            us (shape=(N, nu)): The controls over the trajectory.

        Returns:
            cost_val: The cost of the trajectory.
            new_params: Unused. Included for API compliance.
        """
        xs, xg, xs_err, xf_err, _, _ = self._setup_util(qs, vs, self.qg, self.vg)
        val = 0.5 * (
            self.batch_quadform(xs_err, self.Q) + self.batch_quadform(xf_err, self.Qf) + self.batch_quadform(us, self.R)
        )
        return val, params

    def grad(
        self, qs: jax.Array, vs: jax.Array, us: jax.Array, params: CostFunctionParams
    ) -> Tuple[jax.Array, jax.Array, jax.Array, CostFunctionParams, CostFunctionParams]:
        """Computes the gradient of the cost of a trajectory.

        Args:
            qs (shape=(N + 1, nq)): The generalized positions over the trajectory.
            vs (shape=(N + 1, nv)): The generalized velocities over the trajectory.
            us (shape=(N, nu)): The controls over the trajectory.

        Returns:
            gcost_qs (shape=(N + 1, nq): The gradient of the cost wrt qs.
            gcost_vs (shape=(N + 1, nv): The gradient of the cost wrt vs.
            gcost_us (shape=(N, nu)): The gradient of the cost wrt us.
            gcost_params: Unused. Included for API compliance.
            new_params: Unused. Included for API compliance.
        """
        xs, xg, xs_err, xf_err, nq, _, _ = self._setup_util(qs, vs, self.qg, self.vg)
        gcost_xs = jnp.concatenate(
            (
                self.batch_matmul(xs_err, self.Q),
                (self.Qf @ xf_err)[None, :],
            ),
            axis=-1,
        )
        gcost_qs = gcost_xs[:, :nq]
        gcost_vs = gcost_xs[:, nq:]
        gcost_us = self.batch_matmul(us, self.R)
        return gcost_qs, gcost_vs, gcost_us, params, params

    def hess(
        self, qs: jax.Array, vs: jax.Array, us: jax.Array, params: CostFunctionParams
    ) -> Tuple[jax.Array, jax.Array, jax.Array, CostFunctionParams, CostFunctionParams]:
        """Computes the Hessian of the cost of a trajectory.

        Args:
            qs (shape=(N + 1, nq)): The generalized positions over the trajectory.
            vs (shape=(N + 1, nv)): The generalized velocities over the trajectory.
            us (shape=(N, nu)): The controls over the trajectory.

        Returns:
            Hcost_qs (shape=(N + 1, nq, N + 1, nq)): The Hessian of the cost wrt qs.
                Let t, s be times from 0 to N + 1. Then, d^2/dq_{t,i}dq_{s,j} = Hcost_qs[t, i, s, j].
            Hcost_vs (shape=(N + 1, nv, N + 1, nv)): The Hessian of the cost wrt vs.
                Let t, s be times from 0 to N + 1. Then, d^2/dv_{t,i}dv_{s,j} = Hcost_vs[t, i, s, j].
            Hcost_us (shape=(N, nu, N, nu)): The Hessian of the cost wrt us.
                Let t, s be times from 0 to N. Then, d^2/du_{t,i}du_{s,j} = Hcost_us[t, i, s, j].
            Hcost_params: Unused. Included for API compliance.
            new_params: Unused. Included for API compliance.
        """
        N = us.shape[0]
        xs, xg, xs_err, xf_err, nq, _, _ = self._setup_util(qs, vs, self.qg, self.vg)
        Q_tiled = jnp.tile(self.Q[None, :, None, :], (N + 1, 1, N + 1, 1))
        Hcost_xs = Q_tiled.at[-1, :, -1, :].set(self.Qf)

        Hcost_qs = Hcost_xs[:, :nq, :, :nq]
        Hcost_vs = Hcost_xs[:, nq:, :, nq:]
        Hcost_us = jnp.tile(self.R[None, :, None, :], (N, 1, N, 1))

        return Hcost_qs, Hcost_vs, Hcost_us, params, params
