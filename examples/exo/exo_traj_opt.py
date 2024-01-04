import functools
import os
import pickle
import time

import jax
import mediapy as media
from brax.envs.wrappers.training import DomainRandomizationVmapWrapper
from jax import grad, jacfwd, jit
from jax import numpy as jp
from jax.example_libraries import optimizers

import wandb
from ambersim.envs.exo_base import BehavState, Exo, ExoConfig
from ambersim.envs.exo_parallel import (
    CustomVecEnv,
    rand_friction,
    randomizeBoxTerrain,
    randomizeCoMOffset,
    randomizeSlope,
)
from ambersim.logger.logger import WandbLogger

# XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl python exo_traj_opt.py
NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
    with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
        f.write(
            """{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
"""
        )

# Set environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_GL"] = "egl"


def evalCostImpact(rng, jit_env_reset, jit_env_step, env, alpha, timesteps=400, log_file="traj_opt.log", logger=None):
    """Evaluate Costs from standing position."""
    costs = 0.0
    # step_dur = 1.0
    penalty = 10.0
    survive_reward = 10.0
    P_values = []
    t_values = []
    state = jit_env_reset(rng, BehavState.Walking)
    state.info["alpha"] = state.info["alpha"].at[:].set(alpha)
    init_foot_pos = state.pipeline_state.geom_xpos[:, env.foot_geom_idx[0], 0]
    with open(log_file, "w") as file:
        for _ in range(timesteps):
            # state_change = False
            start_time = time.time()
            state = jit_env_step(state, jp.zeros((rng.shape[0], env.action_size)))
            end_time = time.time()

            step_duration = end_time - start_time
            file.write(f"Step duration: {step_duration}\n")
            P_values.append(state.info["mechanical_power"])
            t_values.append(state.pipeline_state.time)

            if state.info["state"][0] == BehavState.WantToStart:
                state.info["state"] = state.info["state"].at[:].set(BehavState.Walking)

            if sum(state.done):
                costs = costs + penalty * sum(state.done)

            costs = costs - survive_reward * (state.done.shape[0] - sum(state.done))

        step_length = state.pipeline_state.geom_xpos[:, env.foot_geom_idx[0], 0] - init_foot_pos
        mcot = jp.sum(env.mcot(jp.array(t_values).T, step_length, jp.array(P_values).T))
        # set where there's inf or nan mcot to 1000
        mcot = jp.where(jp.isnan(mcot), 1000, mcot)
        mcot = jp.where(jp.isinf(mcot), 1000, mcot)
        costs = costs + mcot
        jax.debug.print("mcot: {}", env.mcot(jp.array(t_values).T, step_length, jp.array(P_values).T))

        value = jp.sum(jp.array(costs)) / (rng.shape[0])

        #print("Value:", value)
        #logger.log_metric("value", value)
        #logger.log_metric("alpha", wandb.Histogram(alpha))

    return value


# Function to evaluate cost
def evalCostFromStanding(rng, state, jit_env_step, env, alpha, timesteps=300, log_file="traj_opt.log", logger=None):
    """Evaluate Costs from standing position."""
    costs = 0.0
    step_dur = 1.0

    # periodic_penalty_weight = 1000000.0
    penalty = 1000.0
    survive_reward = 30.0
    P_values = []
    t_values = []

    # maxErrorTrigger = 0.01
    # minTransitionTime = 0.05

    # breakpoint()

    init_foot_pos = state.pipeline_state.geom_xpos[:, env.foot_geom_idx[0], 0]
    with open(log_file, "w") as file:
        for _ in range(timesteps):
            # state_change = False
            start_time = time.time()
            state = jit_env_step(state, alpha, step_dur)
            end_time = time.time()

            step_duration = end_time - start_time
            file.write(f"Step duration: {step_duration}\n")
            P_values.append(state.info["mechanical_power"])
            t_values.append(state.pipeline_state.time)

            if state.info["state"][0] == BehavState.WantToStart:
                # if state.pipeline_state.time[0] > 0.2:
                # minTransitionTime = 2 * env.step_dur[BehavState.WantToStart]
                # if env.state_condition_met(maxErrorTrigger, minTransitionTime, state.pipeline_state, state.info):
                state.info["state"] = state.info["state"].at[:].set(BehavState.Walking)
                # state_change = True

            elif (
                state.info["state"][0] == BehavState.Walking
                and (state.pipeline_state.time[0] - state.info["domain_info"]["step_start"][0]) / step_dur > 1
            ):
                state.info["domain_info"]["step_start"] = state.pipeline_state.time

                alpha = jp.dot(env.R, alpha)
                print("update step_start")

                q_init = env._forward(
                    state.pipeline_state.time[0], state.info["domain_info"]["step_start"][0], step_dur, alpha
                )
                # q_pos_slice_expanded = jnp.repeat(q_init,state.pipeline_state.qpos.shape[1], axis=1)
                # breakpoint()
                state.info["offset"] = state.pipeline_state.qpos[:, -env.model.nu :] - q_init
                if jp.any(jp.isnan(state.info["offset"])):
                    breakpoint()
                jax.debug.print("offset: {}", state.info["offset"])

                # end_jt = env._forward(1,0,1,alpha)
                # init_next_jt = env._forward(0,0,1,jp.dot(env.R,alpha))
                # periodic_penalty = jp.linalg.norm(end_jt - init_next_jt)
                # cost = cost + periodic_penalty_weight*periodic_penalty

            if sum(state.done):
                costs = costs + penalty * sum(state.done)
                # break

            costs = costs - survive_reward * (state.done.shape[0] - sum(state.done))

        step_length = state.pipeline_state.geom_xpos[:, env.foot_geom_idx[0], 0] - init_foot_pos
        costs = costs + jp.sum(env.mcot(jp.array(t_values).T, step_length, jp.array(P_values).T))
        value = jp.sum(jp.array(costs)) / (state.pipeline_state.time[0] * rng.shape[0])

        logger.log_metric("value", value)
        logger.log_metric("alpha", wandb.Histogram(alpha))
        
    return value


def evalCost(rng, jit_env_reset, jit_env_step, env, alpha, step_dur, timesteps=110, log_file="traj_opt.log", logger=None):
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
                jax.debug.print("done: {}", state.done)
                # break
            # if sum(state.done) or jp.any(jp.isnan(state.info["mcot"])):
            #     costs = costs + penalty * sum(state.done) + penalty * jp.isnan(state.info["mcot"])
            #     # costs.append(sum(state.done)*100000.0)
            #     # costs.append(sum(jp.isnan(state.info["mcot"]))*1000.0)
            #     break
            # else:
        step_length = env.getFootPos(state) - init_foot_pos
        costs = costs + env.mcot(state, t_values, step_length, P_values)
        value = jp.sum(jp.array(costs)) / (i * rng.shape[0])

        logger.log_metric("value", value)
        logger.log_metric("alpha", wandb.Histogram(alpha))

    return value


# Step function for optimization
def step(step, opt_state, get_params, opt_update, evalCostFunc):
    """Optimization step."""
    # breakpoint()
    # value, grads = jax.value_and_grad(evalCostFunc)(get_params(opt_state))
    grad_func = jacfwd(evalCostFunc)
    alpha = get_params(opt_state)
    grads = grad_func(alpha)
    value = evalCostFunc(alpha)
    opt_state = opt_update(step, grads, opt_state)

    return value, opt_state


# Function to load parameters and run a simulation
def load_and_simulate(
    rng, env, jit_env_reset, jit_env_step, params_file, step_dur, num_steps=400, output_video="sim_video1.mp4"
):
    """Load optimized parameters and simulate the environment."""
    with open(params_file, "rb") as file:
        optimized_params = pickle.load(file)
    # TODO: check nominal traj in other setup
    # optimized_params = env.alpha
    print("optimized_params", optimized_params)
    # breakpoint()
    state = jit_env_reset(rng, BehavState.WantToStart)
    images = []
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
config = ExoConfig()
config.slope = True
config.hip_regulation = True
env = Exo(config)

num_env = 20
rng = jax.random.PRNGKey(0)
rng = jax.random.split(rng, num_env)
# domain_env = CustomVecEnv(env, randomization_fn=functools.partial(rand_friction, rng=rng))

domain_env = CustomVecEnv(
    env, randomization_fn=functools.partial(randomizeSlope, rng=rng, plane_ind=0, max_angle_degrees=3)
)

# jit_env_reset = jax.jit(domain_env.reset_bez)
# jit_env_step = jax.jit(domain_env.bez_step)
jit_env_step = jax.jit(domain_env.step)

# vecEnv = ExoParallel(rng=rng)

jit_env_reset = jax.jit(domain_env.reset)
# jit_env_step = jax.jit(env.bez_step)
alpha = env.alpha
step_dur = env.step_dur[BehavState.Walking]

# state = jit_env_reset(rng, BehavState.WantToStart)
state = jit_env_reset(rng, BehavState.Walking)
# state = domain_env.bez_step(state, alpha, step_dur)
# state = jit_env_step(state, alpha, step_dur)
# state = domain_env.bez_step(state, alpha, step_dur)

opt_file = "traj_opt/optimized_params_impact.pkl"
if train_flag:
    alpha = env.alpha
    step_dur = env.step_dur[BehavState.Walking]  # env.step_dur[BehavState.Walking]

    # Initialize optimization
    step_size = 1e-3
    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(alpha)

    num_steps = 5
    cost_func = "evalCostImpact"

    # Initialize wandb logger
    log_dir = None
    project_name = "traj_opt"
    cfg_dict = {
                "cost_func": cost_func,
                "num_steps": num_steps,
                "step_size": step_size,
                }
    
    logger = WandbLogger(log_dir, project_name, cfg_dict)

    costFuncs = {"evalCostImpact": evalCostImpact, "evalCostFromStanding": evalCostFromStanding, "evalCost": evalCost}
    evalCostFunc = functools.partial(costFuncs[cost_func], rng, jit_env_reset, jit_env_step, domain_env, logger=logger)

    # Optimization loop
    for i in range(num_steps):  # Number of optimization steps
        value, opt_state = step(i, opt_state, get_params, opt_update, evalCostFunc)
        alpha = get_params(opt_state)

        #with open(opt_file, "wb") as file:
        #    pickle.dump(get_params(opt_state), file)

        #opt_iter_file = f"traj_opt/optimized_params_impact{i}.pkl"
        #with open(opt_iter_file, "wb") as file:
        #    pickle.dump(get_params(opt_state), file)

        logger.log_metric("value", cost_func)
        logger.log_metric("alpha", wandb.Histogram(alpha))
    
    # Save optimized parameters

else:
    # Load and simulate with optimized parameters

    env = Exo(config)
    # env.getRender()
    # jit_env_reset = jax.jit(env.reset)
    # jit_env_step = jax.jit(env.step)

    print("Loading optimized parameters and simulating...")
    # print alpha
    print(env.alpha)

    # filename = "traj_opt_policy_from_standing_update.mp4"
    with open(opt_file, "rb") as file:
        optimized_params = pickle.load(file)

    print(optimized_params)
    # # breakpoint()
    # env.run_bez_sim_from_standing(
    #     rng=jax.random.PRNGKey(0),
    #     alpha=optimized_params,
    #     step_dur=env.step_dur[BehavState.Walking],
    #     num_steps=400,
    #     output_video=filename,
    # )
    # load_and_simulate(jax.random.PRNGKey(0), env, jit_env_reset, jit_env_step, params_file, env.step_dur[BehavState.Walking])
    filename = "video/traj_opt_impact_Val_inc.mp4"
    env.run_base_sim(
        rng=jax.random.PRNGKey(0),
        alpha=optimized_params,
        num_steps=200,
        output_video=filename,
    )