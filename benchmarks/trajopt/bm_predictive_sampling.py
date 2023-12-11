import timeit

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit
from mujoco import mjx
from mujoco.mjx._src.types import DisableBit

from ambersim.trajopt.cost import CostFunction, StaticGoalQuadraticCost
from ambersim.trajopt.shooting import VanillaPredictiveSampler, VanillaPredictiveSamplerParams
from ambersim.utils.io_utils import load_mjx_model_and_data_from_file


def make_ps(model: mjx.Model, cost_function: CostFunction, nsamples: int) -> VanillaPredictiveSampler:
    """Makes a predictive sampler for this quick and dirty timing script."""
    stdev = 0.01
    ps = VanillaPredictiveSampler(model=model, cost_function=cost_function, nsamples=nsamples, stdev=stdev)
    return ps


if __name__ == "__main__":
    # initializing the model
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

    # initializing the cost function
    cost_function = StaticGoalQuadraticCost(
        Q=jnp.eye(model.nq + model.nv),
        Qf=10.0 * jnp.eye(model.nq + model.nv),
        R=0.01 * jnp.eye(model.nu),
        # qg=jnp.zeros(model.nq).at[6].set(1.0),  # if force_float=True
        qg=jnp.zeros(model.nq),
        vg=jnp.zeros(model.nv),
    )

    # sampler parameters we pass in independent of the number of samples
    key = jax.random.PRNGKey(0)  # random seed for the predictive sampler
    q0 = jnp.zeros(model.nq).at[6].set(1.0)
    v0 = jnp.zeros(model.nv)
    num_steps = 10
    us_guess = jnp.zeros((num_steps, model.nu))
    params = VanillaPredictiveSamplerParams(key=key, q0=q0, v0=v0, us_guess=us_guess)

    nsamples_list = [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 10000]
    runtimes = []
    throughputs = []
    for nsamples in nsamples_list:
        print(f"Running with nsamples={nsamples}...")
        ps = make_ps(model, cost_function, nsamples)
        optimize_fn = jit(ps.optimize)

        # # [DEBUG] profiling with tensorboard
        # qs_star, vs_star, us_star = optimize_fn(params)  # JIT compiling
        # with jax.profiler.trace("/home/albert/tensorboard"):
        #     qs_star, vs_star, us_star = optimize_fn(params)  # after JIT

        def _time_fn(fn=optimize_fn) -> None:
            """Function to time runtime."""
            qs_star, vs_star, us_star = fn(params)
            qs_star.block_until_ready()
            vs_star.block_until_ready()
            us_star.block_until_ready()

        compile_time = timeit.timeit(_time_fn, number=1)
        print(f"  Compile time: {compile_time}")

        num_timing_iters = 100
        time = timeit.timeit(_time_fn, number=num_timing_iters)
        print(f"  Avg. runtime: {time / num_timing_iters}")  # returns TOTAL time, so compute the average ourselves

        runtimes.append(time / num_timing_iters)
        throughputs.append(nsamples / (time / num_timing_iters))

    plt.scatter(np.array(nsamples_list), np.array(runtimes))
    plt.xlabel("number of samples")
    plt.ylabel("runtime (s)")
    plt.title("Predictive Sampling: Number of Samples vs. Runtime")
    plt.xlim([-100, max(nsamples_list) + 100])
    plt.ylim([0, max(runtimes) + 0.01])
    plt.show()

    plt.scatter(np.array(nsamples_list), np.array(throughputs))
    plt.xlabel("number of samples")
    plt.ylabel("samples per second (s)")
    plt.title("Predictive Sampling: Sampling Throughput vs. Number of Samples")
    plt.xlim([-100, max(nsamples_list) + 100])
    plt.ylim([0, max(throughputs) + 10000])
    plt.show()
