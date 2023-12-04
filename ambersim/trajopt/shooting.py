from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import lax, vmap
from mujoco import mjx
from mujoco.mjx import step

from ambersim.trajopt.base import TrajectoryOptimizer, TrajectoryOptimizerParams
from ambersim.trajopt.cost import CostFunction

"""Shooting methods and their derived subclasses."""


# ##### #
# UTILS #
# ##### #


def shoot(m: mjx.Model, q0: jax.Array, v0: jax.Array, us: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Utility function that shoots a model forward given a sequence of control inputs.

    Args:
        m: The model.
        q0: The initial generalized coordinates.
        v0: The initial generalized velocities.
        us: The control inputs.

    Returns:
        qs (shape=(N + 1, nq)): The generalized coordinates.
        vs (shape=(N + 1, nv)): The generalized velocities.
    """
    # initializing the data
    d = mjx.make_data(m)
    d = d.replace(qpos=q0, qvel=v0)  # setting the initial state.
    d = mjx.forward(m, d)  # setting other internal states like acceleration without integrating

    def scan_fn(d, u):
        """Integrates the model forward one step given the control input u."""
        d = d.replace(ctrl=u)
        d = step(m, d)
        x = jnp.concatenate((d.qpos, d.qvel))  # (nq + nv,)
        return d, x

    # scan over the control inputs to get the trajectory.
    _, xs = lax.scan(scan_fn, d, us, length=us.shape[0])
    _qs = xs[:, : m.nq]
    _vs = xs[:, m.nq : m.nq + m.nv]
    qs = jnp.concatenate((q0[None, :], _qs), axis=0)  # (N + 1, nq)
    vs = jnp.concatenate((v0[None, :], _vs), axis=0)  # (N + 1, nv)
    return qs, vs


# ################ #
# SHOOTING METHODS #
# ################ #

# vanilla API


@struct.dataclass
class ShootingParams(TrajectoryOptimizerParams):
    """Parameters for shooting methods."""

    # inputs into the algorithm
    q0: jax.Array  # shape=(nq,) or (?)
    v0: jax.Array  # shape=(nv,) or (?)
    us_guess: jax.Array  # shape=(N, nu) or (?)

    @property
    def N(self) -> int:
        """The number of time steps.

        By default, we assume us_guess represents the ZOH control inputs. However, in the case that it is actually an
        alternate parameterization, we may need to compute N some other way, which requires overwriting this method.
        """
        return self.us_guess.shape[0]


@struct.dataclass
class ShootingAlgorithm(TrajectoryOptimizer):
    """A trajectory optimization algorithm based on shooting methods."""

    def optimize(self, params: ShootingParams) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Optimizes a trajectory using a shooting method.

        Args:
            params: The parameters of the trajectory optimizer.

        Returns:
            qs (shape=(N + 1, nq) or (?)): The optimized trajectory.
            vs (shape=(N + 1, nv) or (?)): The optimized generalized velocities.
            us (shape=(N, nu) or (?)): The optimized controls.
        """
        raise NotImplementedError


# predictive sampling API


@struct.dataclass
class VanillaPredictiveSamplerParams(ShootingParams):
    """Parameters for generic predictive sampling methods."""

    key: jax.Array  # random key for sampling


@struct.dataclass
class VanillaPredictiveSampler(ShootingAlgorithm):
    """A vanilla predictive sampler object.

    The following choices are made:
    (1) the model parameters are fixed, and are therefore a field of this dataclass;
    (2) the control sequence is parameterized as a ZOH sequence instead of a spline;
    (3) the control inputs are sampled from a normal distribution with a uniformly chosen noise scale over all params.
    (4) the cost function is quadratic in the states and controls.
    """

    model: mjx.Model
    cost_function: CostFunction
    nsamples: int = struct.field(pytree_node=False)
    stdev: float = struct.field(pytree_node=False)  # noise scale, parameters theta_new ~ N(theta, (stdev ** 2) * I)

    def optimize(self, params: VanillaPredictiveSamplerParams) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Optimizes a trajectory using a vanilla predictive sampler.

        Args:
            params: The parameters of the trajectory optimizer.

        Returns:
            qs (shape=(N + 1, nq)): The optimized trajectory.
            vs (shape=(N + 1, nv)): The optimized generalized velocities.
            us (shape=(N, nu)): The optimized controls.
        """
        # unpack the params
        m = self.model
        nsamples = self.nsamples
        stdev = self.stdev

        q0 = params.q0
        v0 = params.v0
        us_guess = params.us_guess
        N = params.N
        key = params.key

        # sample over the control inputs
        _us_samples = us_guess + jax.random.normal(key, shape=(nsamples, N, m.nu)) * stdev

        # clamping the samples to their control limits
        # TODO(ahl): write a create classmethod that allows the user to set default_limits optionally with some semi-
        # reasonable default value
        # TODO(ahl): check whether joints with no limits have reasonable defaults for m.actuator_ctrlrange
        limits = m.actuator_ctrlrange
        clip_fn = partial(jnp.clip, a_min=limits[:, 0], a_max=limits[:, 1])  # clipping function with limits already set
        us_samples = vmap(vmap(clip_fn))(_us_samples)  # apply limits only to the last dim, need a nested vmap
        # limited = m.actuator_ctrllimited[:, None]  # (nu, 1) whether each actuator has limited control authority
        # default_limits = jnp.array([[-1000.0, 1000.0]] * m.nu)  # (nu, 2) default limits for each actuator
        # limits = jnp.where(limited, m.actuator_ctrlrange, default_limits)  # (nu, 2)

        # predict many samples, evaluate them, and return the best trajectory tuple
        # vmap over the input data and the control trajectories
        qs_samples, vs_samples = vmap(shoot, in_axes=(None, None, None, 0))(m, q0, v0, us_samples)
        costs, _ = vmap(self.cost_function.cost, in_axes=(0, 0, 0, None))(
            qs_samples, vs_samples, us_samples, None
        )  # (nsamples,)
        best_idx = jnp.argmin(costs)
        qs_star = lax.dynamic_slice(qs_samples, (best_idx, 0, 0), (1, N + 1, m.nq))[0]  # (N + 1, nq)
        vs_star = lax.dynamic_slice(vs_samples, (best_idx, 0, 0), (1, N + 1, m.nv))[0]  # (N + 1, nv)
        us_star = lax.dynamic_slice(us_samples, (best_idx, 0, 0), (1, N, m.nu))[0]  # (N, nu)
        return qs_star, vs_star, us_star
