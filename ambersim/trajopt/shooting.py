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


def shoot(m: mjx.Model, x0: jax.Array, us: jax.Array) -> jax.Array:
    """Utility function that shoots a model forward given a sequence of control inputs.

    Args:
        m: The model.
        x0: The initial state.
        us: The control inputs.

    Returns:
        xs (shape=(N + 1, nq + nv)): The state trajectory.
    """
    # initializing the data
    d = mjx.make_data(m)
    d = d.replace(qpos=x0[: m.nq], qvel=x0[m.nq :])  # setting the initial state.
    d = mjx.forward(m, d)  # setting other internal states like acceleration without integrating

    def scan_fn(d, u):
        """Integrates the model forward one step given the control input u."""
        d = d.replace(ctrl=u)
        d = step(m, d)
        x = jnp.concatenate((d.qpos, d.qvel))  # (nq + nv,)
        return d, x

    # scan over the control inputs to get the trajectory.
    _, _xs = lax.scan(scan_fn, d, us, length=us.shape[0])
    xs = jnp.concatenate((x0[None, :], _xs), axis=0)  # (N + 1, nq + nv)
    return xs


# ################ #
# SHOOTING METHODS #
# ################ #

# vanilla API


@struct.dataclass
class ShootingParams(TrajectoryOptimizerParams):
    """Parameters for shooting methods."""

    # inputs into the algorithm
    x0: jax.Array  # shape=(nq + nv,) or (?)
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

    def optimize(self, params: ShootingParams) -> Tuple[jax.Array, jax.Array]:
        """Optimizes a trajectory using a shooting method.

        Args:
            params: The parameters of the trajectory optimizer.

        Returns:
            xs_star (shape=(N + 1, nq) or (?)): The optimized trajectory.
            us_star (shape=(N, nu) or (?)): The optimized controls.
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

    def optimize(self, params: VanillaPredictiveSamplerParams) -> Tuple[jax.Array, jax.Array]:
        """Optimizes a trajectory using a vanilla predictive sampler.

        Args:
            params: The parameters of the trajectory optimizer.

        Returns:
            xs (shape=(N + 1, nq + nv)): The optimized trajectory.
            us (shape=(N, nu)): The optimized controls.
        """
        # unpack the params
        m = self.model
        nsamples = self.nsamples
        stdev = self.stdev

        x0 = params.x0
        us_guess = params.us_guess
        N = params.N
        key = params.key

        # sample over the control inputs - the first sample is the guess, since it's possible that it's the best one
        noise = jnp.concatenate(
            (jnp.zeros((1, N, m.nu)), jax.random.normal(key, shape=(nsamples - 1, N, m.nu)) * stdev), axis=0
        )
        _us_samples = us_guess + noise

        # clamping the samples to their control limits
        limits = m.actuator_ctrlrange
        clip_fn = partial(jnp.clip, a_min=limits[:, 0], a_max=limits[:, 1])  # clipping function with limits already set
        us_samples = vmap(vmap(clip_fn))(_us_samples)  # apply limits only to the last dim, need a nested vmap

        # predict many samples, evaluate them, and return the best trajectory tuple
        # vmap over the input data and the control trajectories
        xs_samples = vmap(shoot, in_axes=(None, None, 0))(m, x0, us_samples)
        costs, _ = vmap(self.cost_function.cost, in_axes=(0, 0, None))(xs_samples, us_samples, None)  # (nsamples,)
        best_idx = jnp.argmin(costs)
        xs_star = lax.dynamic_slice(xs_samples, (best_idx, 0, 0), (1, N + 1, m.nq + m.nv))[0]  # (N + 1, nq + nv)
        us_star = lax.dynamic_slice(us_samples, (best_idx, 0, 0), (1, N, m.nu))[0]  # (N, nu)
        return xs_star, us_star
