import timeit

import jax
import jax.numpy as jnp
from jax import jit
from mujoco.mjx._src.types import DisableBit

from ambersim.trajopt.cost import StaticGoalQuadraticCost
from ambersim.trajopt.shooting import VanillaPredictiveSampler, VanillaPredictiveSamplerParams
from ambersim.utils.io_utils import load_mjx_model_and_data_from_file

if __name__ == "__main__":
    # initializing the predictive sampler
    model, _ = load_mjx_model_and_data_from_file("models/barrett_hand/bh280.xml", force_float=False)
    model = model.replace(
        opt=model.opt.replace(
            timestep=0.002,  # dt
            iterations=1,  # number of Newton steps to take during solve
            ls_iterations=4,  # number of line search iterations along step direction
            integrator=0,  # Euler semi-implicit integration
            solver=2,  # Newton solver
            disableflags=DisableBit.CONTACT,  # [IMPORTANT] disable contact for this example
        )
    )
    cost_function = StaticGoalQuadraticCost(
        Q=jnp.eye(model.nq + model.nv),
        Qf=10.0 * jnp.eye(model.nq + model.nv),
        R=0.01 * jnp.eye(model.nu),
        # xg=jnp.zeros(model.nq + model.nv).at[6].set(1.0),  # if force_float=True
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

    # [DEBUG] profiling with nsight systems
    # xs_star, us_star = optimize_fn(params)  # JIT compiling
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    #     xs_star us_star = optimize_fn(params)  # after JIT

    def _time_fn():
        xs_star, us_star = optimize_fn(params)
        xs_star.block_until_ready()
        us_star.block_until_ready()

    compile_time = timeit.timeit(_time_fn, number=1)
    print(f"Compile time: {compile_time}")

    # informal timing test
    # TODO(ahl): identify bottlenecks and zap them
    # [Dec. 3, 2023] on vulcan, I've informally tested the scaling of runtime with the number of steps and the number
    # of samples. Here are a few preliminary results:
    # * nsamples=100, numsteps=10. avg: 0.01s
    # * nsamples=1000, numsteps=10. avg: 0.015s
    # * nsamples=10000, numsteps=10. avg: 0.07s
    # * nsamples=100, numsteps=100. avg: 0.1s
    # we conclude that the runtime scales predictably linearly with numsteps, but we also have some sort of (perhaps
    # logarithmic) scaling of runtime with nsamples. this outlook is somewhat grim, and we need to also keep in mind
    # that we've completely disabled contact for this example and set the number of solver iterations and line search
    # iterations to very runtime-friendly values
    num_timing_iters = 100
    time = timeit.timeit(_time_fn, number=num_timing_iters)
    print(f"Avg. runtime: {time / num_timing_iters}")  # timeit returns TOTAL time, so we compute the average ourselves
    breakpoint()
