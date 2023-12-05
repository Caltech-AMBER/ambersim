import jax
import jax.numpy as jnp
from jax import jit
from mujoco.mjx._src.types import DisableBit

from ambersim.trajopt.cost import StaticGoalQuadraticCost
from ambersim.trajopt.shooting import VanillaPredictiveSampler, VanillaPredictiveSamplerParams
from ambersim.utils.io_utils import load_mjx_model_and_data_from_file


def test_smoke_VPS():
    """Simple smoke test to make sure we can run inputs through the vanilla predictive sampler + jit."""
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

    # sampler parameters
    key = jax.random.PRNGKey(0)  # random seed for the predictive sampler
    # x0 = jnp.zeros(model.nq + model.nv).at[6].set(1.0)  # if force_float=True
    x0 = jnp.zeros(model.nq + model.nv)
    num_steps = 10
    us_guess = jnp.zeros((num_steps, model.nu))
    params = VanillaPredictiveSamplerParams(key=key, x0=x0, us_guess=us_guess)

    # sampling the best sequence of qs, vs, and us
    optimize_fn = jit(ps.optimize)
    assert optimize_fn(params)
