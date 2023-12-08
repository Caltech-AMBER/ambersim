import jax
import jax.numpy as jnp
from jax import jit, lax
from mujoco import mjx

from ambersim.control.predictive_control import PredictiveSamplingController, PredictiveSamplingControllerParams

"""This file contains simulation utils for various controllers."""


def simulate_predictive_sampling_controller(
    model: mjx.Model,
    controller: PredictiveSamplingController,
    x0: jax.Array,
    num_steps: int,
    N: int,
    seed: int = 0,
    physics_steps_per_control_step: int = 1,
) -> None:
    """Simulates a closed-loop system.

    Args:
        model: The model. Properties like the timestep are set in here.
        controller: The controller.
        x0: The initial state.
        num_steps: The number of steps to simulate for.
        seed: The random seed.
        physics_steps_per_control_step: The number of physics steps to take per control step.

    Returns:
        data: The internal data of the model after the simulation.
        xs: The state trajectory.
    """
    # initial setup
    print("Initial setup...")
    dt = model.opt.timestep

    data = mjx.make_data(model)
    data = data.replace(qpos=x0[: model.nq], qvel=x0[model.nq :])  # setting the initial state.
    data = mjx.forward(model, data)  # setting other internal states like acceleration without integrating

    xs_hist = jnp.zeros((num_steps + 1, model.nq + model.nv))
    xs_hist = xs_hist.at[0, :].set(x0)

    us_hist = jnp.zeros((num_steps, model.nu))
    key = jax.random.PRNGKey(seed)

    jit_compute = jit(
        lambda key, x_meas, us_guess: controller.compute_with_us_star(
            PredictiveSamplingControllerParams(key=key, x=x_meas, us_guess=us_guess)
        )
    )
    jit_step = jit(lambda data, u: mjx.step(model, data.replace(ctrl=u)))

    us_guess = jnp.zeros((N, model.nu))
    import time

    times = []
    for i in range(num_steps // physics_steps_per_control_step):
        start = time.time()
        # computing the control input
        x_meas = jnp.concatenate((data.qpos, data.qvel))
        if i == 0:
            print("Compiling the compute function...")
        u, us_guess = jit_compute(key, x_meas, us_guess)
        if i == 0:
            print("Compiled the compute function!")

        # simulating the system forward
        for j in range(physics_steps_per_control_step):
            print(f"t: {(i * physics_steps_per_control_step + j) * dt:.2f} | u: {u}")
            if i == 0 and j == 0:
                print("Compiling the step function...")
            breakpoint()
            data = jit_step(data, u)  # <-- segfaults here! non-jitted version doesn't segfault
            if i == 0 and j == 0:
                print("Compiled the step function!")
            us_hist = us_hist.at[i, :].set(u)
            xs_hist = xs_hist.at[i + 1, :].set(jnp.concatenate((data.qpos, data.qvel)))

        key = jax.random.split(key)[0]
        end = time.time()
        times.append(end - start)
    return data, xs_hist, us_hist, times  # [DEBUG] return iteration times


if __name__ == "__main__":
    # TODO(ahl): move this into a proper script
    import time

    import mujoco
    import mujoco.viewer
    import numpy as np
    from mujoco.mjx._src.types import DisableBit

    from ambersim.trajopt.cost import StaticGoalQuadraticCost
    from ambersim.trajopt.shooting import VanillaPredictiveSampler
    from ambersim.utils.io_utils import load_mj_model_from_file, mj_to_mjx_model_and_data

    # loading model + defining the predictive controller
    # mj_model = load_mj_model_from_file("models/barrett_hand/bh280.xml")
    mj_model = load_mj_model_from_file("models/allegro_hand/right_hand.xml")
    mj_model.opt.timestep = 0.003
    model, _ = mj_to_mjx_model_and_data(mj_model)
    model = model.replace(
        opt=model.opt.replace(
            timestep=0.003,  # dt
            iterations=1,  # number of Newton steps to take during solve
            ls_iterations=4,  # number of line search iterations along step direction
            integrator=0,  # Euler semi-implicit integration
            solver=2,  # Newton solver
            disableflags=DisableBit.CONTACT,  # disable contact for this test
        )
    )
    cost_function = StaticGoalQuadraticCost(
        Q=jnp.diag(jnp.array([10.0] * model.nq + [1.0] * model.nv)),
        Qf=jnp.diag(jnp.array([10.0] * model.nq + [1.0] * model.nv)),
        R=jnp.eye(model.nu),
        xg=jnp.zeros(model.nq + model.nv),
    )
    nsamples = 1  # N=10, dt=0.003, nsamples=1000 just barely hits realtime rates
    stdev = 0.25
    ps = VanillaPredictiveSampler(model=model, cost_function=cost_function, nsamples=nsamples, stdev=stdev)

    N = 10
    controller = PredictiveSamplingController(trajectory_optimizer=ps, model=model)

    print("Controller created! Simulating...")

    # simulating forward
    key = jax.random.PRNGKey(1234)
    # x0 = jnp.array([0.14021026, 0.04142465, 0.99314054, -0.00491294, -0.00487147,  0.81353587, -0.00491293, -0.00487143, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x0 = 2.0 * (jax.random.uniform(key, shape=(model.nq + model.nv,)) - 0.5)
    T = 3.0
    num_steps = int(T / model.opt.timestep)
    data, xs, us, times = simulate_predictive_sampling_controller(
        model, controller, x0, num_steps, N, seed=1337, physics_steps_per_control_step=N
    )  # just testing warm starting the sim
    print(np.mean(times[1:]))

    # post-hoc visualization
    i = 0
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = x0[: model.nq]
    mj_data.qvel[:] = x0[model.nq :]
    dt = model.opt.timestep
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            start = time.time()
            ui = us[i, :]
            mj_data.ctrl[:] = ui
            if i <= int(T / dt):
                mujoco.mj_step(mj_model, mj_data)
                print(f"t: {i * dt:.2f} | qpos: {mj_data.qpos}")
                print(f"t: {i * dt:.2f} | cost: {mj_data.qpos @ mj_data.qpos + mj_data.qvel @ mj_data.qvel:.2f}")
                viewer.sync()
                i += 1
                elapsed = time.time() - start
                if elapsed < mj_model.opt.timestep:
                    time.sleep(mj_model.opt.timestep)

    breakpoint()
