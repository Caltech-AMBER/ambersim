import functools
import os
import pickle
import time

import jax
import mediapy as media
import wandb
from brax.envs.wrappers.training import DomainRandomizationVmapWrapper
from jax import grad, jacfwd, jit
from jax import numpy as jp
from jax.example_libraries import optimizers

from ambersim.envs.exo_base import BehavState, Exo
from ambersim.envs.exo_parallel import CustomVecEnv, ExoParallel, rand_friction, randomizeCoMOffset

# Set environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Function to evaluate cost
def evalCostFromStanding(rng, state, jit_env_step, env, alpha, timesteps=110, log_file="traj_opt.log"):
    """Evaluate Costs from standing position."""
    costs = 0.0
    step_dur = 1.0

    # periodic_penalty_weight = 1000000.0
    penalty = 100000.0
    P_values = []
    t_values = []

    maxErrorTrigger = 0.01
    minTransitionTime = 0.05

    # breakpoint()

    # init_foot_pos = env.getFootPos(state)
    with open(log_file, "w") as file:
        for _ in range(timesteps):
            start_time = time.time()
            state = jit_env_step(state, alpha, step_dur)
            end_time = time.time()

            step_duration = end_time - start_time
            file.write(f"Step duration: {step_duration}\n")
            P_values.append(state.info["mechanical_power"])
            t_values.append(state.pipeline_state.time[0])

            if state.info["state"] == BehavState.WantToStart:
                minTransitionTime = 2 * env.step_dur[BehavState.WantToStart]
                if env.state_condition_met(maxErrorTrigger, minTransitionTime, state.pipeline_state, state.info):
                    state.info["state"] = BehavState.Walking
                    state_change = True

            elif (
                state.info["state"] == BehavState.Walking
                and (state.pipeline_state.time - state.info["domain_info"]["step_start"]) / step_dur > 1
            ):
                state.info["domain_info"]["step_start"] = state.pipeline_state.time

                alpha = jp.dot(env.R, alpha)
                print("update step_start")

                state.info["offset"] = state.pipeline_state.qpos[-env.model.nu :] - env._forward(
                    state.pipeline_state.time, state.info["domain_info"]["step_start"], step_dur, alpha
                )
                jax.debug.print("offset: {}", state.info["offset"])

                # end_jt = env._forward(1,0,1,alpha)
                # init_next_jt = env._forward(0,0,1,jp.dot(env.R,alpha))
                # periodic_penalty = jp.linalg.norm(end_jt - init_next_jt)
                # cost = cost + periodic_penalty_weight*periodic_penalty

            if state_change:
                state.info["domain_info"]["step_start"] = state.pipeline_state.time
                state.info["offset"] = state.pipeline_state.qpos[-env.model.nu :] - env.getNominalDesire(state)[0]

            if sum(state.done):
                costs = costs + penalty * sum(state.done)
                break

        # step_length = env.getFootPos(state) - init_foot_pos
        # costs = costs + env.mcot(state, t_values, step_length, P_values)

    return jp.sum(jp.array(costs)) / (i * rng.shape[0])


def evalCost(rng, jit_env_reset, jit_env_step, env, alpha, step_dur, timesteps=110, log_file="traj_opt.log"):
    """Evaluate Costs Walking Only."""
    state = jit_env_reset(rng, alpha, step_dur)
    costs = 0.0

    periodic_penalty_weight = 1000000.0
    penalty = 100000.0
    P_values = []
    t_values = []

    init_foot_pos = env.getFootPos(state)
    with open(log_file, "w") as file:
        for _ in range(timesteps):
            start_time = time.time()
            state = jit_env_step(state, alpha, step_dur)
            end_time = time.time()

            step_duration = end_time - start_time
            file.write(f"Step duration: {step_duration}\n")
            P_values.append(state.info["mechanical_power"])
            t_values.append(state.pipeline_state.time[0])
            if (state.pipeline_state.time[0] - state.info["domain_info"]["step_start"][0]) / step_dur > 1:
                state.info["domain_info"]["step_start"] = state.pipeline_state.time

                alpha = jp.dot(env.R, alpha)
                print("update step_start")

                end_jt = env._forward(1, 0, 1, alpha)
                init_next_jt = env._forward(0, 0, 1, jp.dot(env.R, alpha))
                periodic_penalty = jp.linalg.norm(end_jt - init_next_jt)
                costs = costs + periodic_penalty_weight * periodic_penalty

            if sum(state.done):
                costs = costs + penalty * sum(state.done)
                break
            # if sum(state.done) or jp.any(jp.isnan(state.info["mcot"])):
            #     costs = costs + penalty * sum(state.done) + penalty * jp.isnan(state.info["mcot"])
            #     # costs.append(sum(state.done)*100000.0)
            #     # costs.append(sum(jp.isnan(state.info["mcot"]))*1000.0)
            #     break
            # else:
        step_length = env.getFootPos(state) - init_foot_pos
        costs = costs + env.mcot(state, t_values, step_length, P_values)

    return jp.sum(jp.array(costs)) / (i * rng.shape[0])


# Step function for optimization
def step(step, opt_state, get_params, opt_update, evalCostFunc):
    """Optimization step."""
    value, grads = jax.value_and_grad(evalCostFunc)(get_params(opt_state))
    opt_state = opt_update(step, grads, opt_state)
    return value, opt_state


# Function to load parameters and run a simulation
def load_and_simulate(
    rng, env, jit_env_reset, jit_env_step, params_file, step_dur, num_steps=400, output_video="sim_video1.mp4"
):
    """Load optimized parameters and simulate the environment."""
    # with open(params_file, 'rb') as file:
    #     optimized_params = pickle.load(file)
    # TODO: check nominal traj in other setup
    optimized_params = env.alpha
    state = jit_env_reset(rng, BehavState.WantToStart)
    images = []
    # breakpoint()
    env.getRender()
    for _ in range(num_steps):
        state = jit_env_step(state, optimized_params, step_dur)
        if (state.pipeline_state.time - state.info["domain_info"]["step_start"]) / step_dur > 1:
            state.info["domain_info"]["step_start"] = state.pipeline_state.time
            optimized_params = jp.dot(env.R, optimized_params)
            print("update step_start")
        images.append(env.get_image(state.pipeline_state))
    media.write_video(output_video, images, fps=1.0 / env.dt)


# Training flag
train_flag = True

# Environment setup
env = Exo()

num_env = 25
rng = jax.random.PRNGKey(0)
rng = jax.random.split(rng, num_env)
domain_env = CustomVecEnv(env, randomization_fn=functools.partial(rand_friction, rng=rng))

# jit_env_reset = jax.jit(env.reset_bez)
# jit_env_step = jax.jit(env.bez_step)

# vecEnv = ExoParallel(rng=rng)

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.bez_step)
alpha = env.alpha
step_dur = env.step_dur[BehavState.Walking]

state = jit_env_reset(jax.random.PRNGKey(0), BehavState.WantToStart)

if train_flag:
    alpha = env.alpha
    step_dur = 1.0  # env.step_dur[BehavState.Walking]

    evalCostFunc = functools.partial(evalCostFromStanding, jax.random.PRNGKey(0), state, jit_env_step, env)

    # Initialize optimization
    opt_init, opt_update, get_params = optimizers.adam(1e-20)
    opt_state = opt_init(alpha)

    # Initialize wandb run
    num_steps = 5
    wandb.init(
        project="traj_opt",
        config={
            "num_steps": num_steps,
            "step_size": 1e-8,
        },
    )

    # Optimization loop
    for i in range(num_steps):  # Number of optimization steps
        value, opt_state = step(i, opt_state, get_params, opt_update, evalCostFunc)
        alpha = get_params(opt_state)

        # Log results with wandb
        wandb.log({"step": i, "value": value, "alpha": wandb.Histogram(alpha)})

    # Save optimized parameters
    with open("optimized_params.pkl", "wb") as file:
        pickle.dump(get_params(opt_state), file)
else:
    # Load and simulate with optimized parameters
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.bez_step)
    params_file = "optimized_params.pkl"
    load_and_simulate(rng[0], env, jit_env_reset, jit_env_step, params_file, env.step_dur)


# import functools
# import jax
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# # Disable memory preallocation
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# import time
# import mediapy as media
# from jax import numpy as jp
# import pickle

# # Configure MuJoCo to use the EGL rendering backend (requires GPU)
# # print('Setting environment variable to use GPU rendering:')

# from jax import grad, jit, random, jacfwd
# from jax.example_libraries import optimizers
# import wandb
# from ambersim.envs.exo import Exo

# #TODO: add domain randomization, add terminal penaly, modify cost function

# def evalCost(jit_env_reset, jit_env_step, env,alpha,step_dur, timesteps = 100):
#     state = jit_env_reset(rng,alpha,step_dur)
#     costs = []
#     for _ in range(timesteps):
#         start = time.time()
#         # rollout.append(state.pipeline_state)

#         state = jit_env_step(state,alpha,step_dur)
#         costs.append(state.info["mcot"])

#         if (state.pipeline_state.time - state.info["domain_info"]["step_start"]) / step_dur > 1:
#             state.info["domain_info"]["step_start"] = state.pipeline_state.time
#             alpha = jp.dot(env.R, alpha)
#             print(f"update step_start")

#         end = time.time()
#         print(f"step time: {end - start}")
#     return jp.sum(jp.array(costs))


# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
# env = Exo()

# jit_env_reset = jax.jit(env.reset_bez)
# jit_env_step = jax.jit(env.bez_step)

# train_flag = False

# if train_flag:
#     alpha = env.alpha
#     step_dur = env.step_dur
#     # warming up JIT-compiled function
#     print("warming up JIT compilation...")
#     rng = jax.random.PRNGKey(0)
#     # rng = jax.random.split(rng, num_env)

#     start = time.time()
#     state = jit_env_reset(rng,alpha,step_dur)

#     end = time.time()
#     print(f"compilation time: {end - start}")

#     evalCostFunc = functools.partial(evalCost,jit_env_reset, jit_env_step, env, step_dur = env.step_dur)

#     grad_func = jacfwd(evalCostFunc)
#     def step(step, opt_state):
#         value, grads = jax.value_and_grad(evalCostFunc)(get_params(opt_state))
#         opt_state = opt_update(step, grads, opt_state)
#         return value, opt_state


#     # Initialize wandb run
#     wandb.init(project="traj_opt", config={
#         "num_steps": 10,
#         "step_size": 1e-7,
#         # Add other configurations if needed
#     })

#     num_steps = wandb.config.num_steps
#     opt_init, opt_update, get_params = optimizers.adam(wandb.config.step_size)

#     # Create a state tuple for storing the optimizer state and parameters
#     opt_state = opt_init((env.alpha))


#     values = []
#     opt_states = []

#     for i in range(num_steps):
#         value, opt_state = step(i, opt_state)
#         values.append(value)
#         opt_states.append(opt_state)

#         # Get current parameters and gradients
#         alpha = get_params(opt_state)
#         grads = grad_func(alpha)

#         # Logging the value, gradients, parameters, and step information to wandb
#         wandb.log({
#             "step": i,
#             "value": value,
#             "gradients": wandb.Histogram(grads),
#             "alpha": wandb.Histogram(alpha),
#         })

#     params_updated = get_params(opt_state)
#     print("Updated alpha:", params_updated)

#     # Optionally, saving the final parameters as an artifact


#     # Save the object
#     with open('optimized_params.pkl', 'wb') as file:
#         pickle.dump(params_updated, file)

#     print("File saved as 'optimized_params.pkl'")

#     artifact = wandb.Artifact('optimized_params', type='model')
#     artifact.add_file('optimized_params.pkl')
#     wandb.log_artifact(artifact)

#     wandb.finish()

# else:


#     def load_and_simulate(env, jit_env_reset, jit_env_step, params_file, step_dur, num_steps=200,output_video = "sim_video.mp4"):
#         # Load optimized parameters
#         with open(params_file, 'rb') as file:
#             optimized_params = pickle.load(file)

#         rng = jax.random.PRNGKey(0)  # Initialize random key
#         state = jit_env_reset(rng, optimized_params, step_dur)  # Reset environment with optimized params
#         images = []
#         env.getRender()
#         # Simulate the environment with the loaded parameters
#         #maybe overwrite the env one and just use that?
#         #need o check logic
#         for _ in range(num_steps):
#             start = time.time()
#             state = jit_env_step(state, optimized_params, step_dur)
#             end = time.time()
#             print(f"Simulation step time: {end - start}")
#             images.append(env.get_image(state.pipeline_state))

#         print("Simulation completed.")
#         media.write_video(output_video, images, fps=1.0 / env.dt)

#     # Example usage
#     env = Exo()  # Initialize your environment
#     jit_env_reset = jax.jit(env.reset_bez)
#     jit_env_step = jax.jit(env.bez_step)

#     params_file = 'optimized_params.pkl'  # Path to your saved parameters file
#     load_and_simulate(env, jit_env_reset, jit_env_step, params_file, env.step_dur)
