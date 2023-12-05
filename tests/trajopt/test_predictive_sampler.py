import os

import jax
import jax.numpy as jnp
import pytest
from jax import jit, vmap
from mujoco.mjx._src.types import DisableBit

from ambersim.trajopt.base import CostFunctionParams
from ambersim.trajopt.cost import StaticGoalQuadraticCost
from ambersim.trajopt.shooting import VanillaPredictiveSampler, VanillaPredictiveSamplerParams, shoot
from ambersim.utils.io_utils import load_mjx_model_and_data_from_file

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # fixes OOM error


@pytest.fixture
def vps_data():
    """Makes data required for testing vanilla predictive sampling."""
    # initializing the predictive sampler
    model, _ = load_mjx_model_and_data_from_file("models/barrett_hand/bh280.xml", force_float=False)
    model = model.replace(
        opt=model.opt.replace(
            timestep=0.002,  # dt
            iterations=1,  # number of Newton steps to take during solve
            ls_iterations=4,  # number of line search iterations along step direction
            integrator=0,  # Euler semi-implicit integration
            solver=2,  # Newton solver
            disableflags=DisableBit.CONTACT,  # disable contact for this test
        )
    )
    cost_function = StaticGoalQuadraticCost(
        Q=jnp.eye(model.nq + model.nv),
        Qf=10.0 * jnp.eye(model.nq + model.nv),
        R=0.01 * jnp.eye(model.nu),
        xg=jnp.zeros(model.nq + model.nv),
    )
    nsamples = 100
    stdev = 0.01
    ps = VanillaPredictiveSampler(model=model, cost_function=cost_function, nsamples=nsamples, stdev=stdev)
    return ps, model, cost_function


def test_smoke_VPS(vps_data):
    """Simple smoke test to make sure we can run inputs through the vanilla predictive sampler + jit."""
    ps, model, _ = vps_data

    # sampler parameters
    key = jax.random.PRNGKey(0)  # random seed for the predictive sampler
    x0 = jnp.zeros(model.nq + model.nv)
    num_steps = 10
    us_guess = jnp.zeros((num_steps, model.nu))
    params = VanillaPredictiveSamplerParams(key=key, x0=x0, us_guess=us_guess)

    # sampling the best sequence of qs, vs, and us
    optimize_fn = jit(ps.optimize)
    assert optimize_fn(params)


def test_VPS_cost_decrease(vps_data):
    """Tests to make sure vanilla predictive sampling decreases (or maintains) the cost."""
    # set up sampler and cost function
    ps, model, cost_function = vps_data

    # batched sampler parameters
    batch_size = 10
    key = jax.random.PRNGKey(0)  # random seed for the predictive sampler
    x0 = jax.random.normal(key=key, shape=(batch_size, model.nq + model.nv))

    key, subkey = jax.random.split(key)
    num_steps = 10
    us_guess = jax.random.normal(key=subkey, shape=(batch_size, num_steps, model.nu))

    keys = jax.random.split(key, num=batch_size)
    params = VanillaPredictiveSamplerParams(key=keys, x0=x0, us_guess=us_guess)

    # sampling with the vanilla predictive sampler
    xs_stars, us_stars = vmap(ps.optimize)(params)

    # "optimal" rollout from predictive sampling
    vmap_cost = jit(vmap(lambda xs, us: cost_function.cost(xs, us, CostFunctionParams())[0], in_axes=(0, 0)))
    costs_star = vmap_cost(xs_stars, us_stars)

    # simply shooting the random initial guess
    xs_guess = vmap(shoot, in_axes=(None, 0, 0))(model, x0, us_guess)
    costs_guess = vmap_cost(xs_guess, us_guess)
    assert jnp.all(costs_star <= costs_guess)
