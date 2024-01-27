import os
import pickle
from itertools import product
from typing import List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from brax import math

from ambersim.envs.exo_base import BehavState, Exo, ExoConfig
from ambersim.envs.exo_parallel import randomizeSlopeGain
from ambersim.envs.exo_traj_opt import TrajectoryOptimizer, TrajOptConfig


def generate_traj_opt_configs(
    num_envs: List[int], num_steps: List[int], opt_step_sizes: List[float]
) -> List[TrajOptConfig]:
    """Generate a list of TrajOptConfig instances based on specified ranges."""
    configs = []
    for num_env, num_step, opt_step_size in product(num_envs, num_steps, opt_step_sizes):
        config = TrajOptConfig(num_env=num_env, num_steps=num_step, opt_step_size=opt_step_size)
        configs.append(config)
    return configs


config = ExoConfig()
config.slope = True
config.traj_opt = True
config.impact_based_switching = True
config.jt_traj_file = "optimized_params.yaml"
config.physics_steps_per_control_step = 5
config.reset_noise_scale = 1e-2
config.no_noise = False
config.controller.hip_regulation = False
config.controller.cop_regulation = True
env = Exo(config)
# param_keys = ["alpha"]
# param_values = [env.alpha]  # Example initial conditions
param_keys = ["cop_regulator_gain"]
param_values = [env.config.controller.cop_regulator_gain]
# import jax.numpy as jp
# cop_gain =  jp.array([[-0.01, -0.0096,0.0], [0.005,-0.0004, 0.0]])


# to do add grf penalty, and check the impact_mismatch

cost_terms = [
    "base_smoothness_reward",
    "tracking_err",
    "tracking_pos_reward",
    "tracking_lin_vel_reward",
    "tracking_ang_vel_reward",
    "survival",
]
cost_weights = {
    "base_smoothness_reward": 0.001,
    "tracking_err": 1.0,
    "tracking_pos_reward": 1.0,
    "tracking_lin_vel_reward": 10.0,
    "tracking_ang_vel_reward": 1.0,
    "survival": 100.0,
}

traj_opt = TrajOptConfig()
traj_opt.num_steps = 30
traj_opt.seed = 1
# loop through different num env values;
num_envs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
paths = {}
base_dir = "traj_results_cop_mod_reward"
dir_yaml = os.path.join(base_dir, "num_envs_dir.yaml")

train = False
if train:
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for num_env in num_envs:
        traj_opt.num_env = num_env
        optimizer = TrajectoryOptimizer(config, traj_opt, param_keys, param_values, cost_terms, cost_weights)

        path = optimizer.save_config(base_dir=base_dir)
        print(f"Config saved at: {path}")
        optimizer.train()

        paths["num_env" + str(num_env)] = path

        # save the paths to a yaml file
        with open(dir_yaml, "w") as file:
            documents = yaml.dump(paths, file)

    # # Sort the paths dictionary by keys (convert keys to integers for sorting)
    # sorted_paths = {k: paths[k] for k in sorted(paths, key=lambda x: int(x))}

    # # Save the sorted paths to a yaml file with pretty formatting
    # with open(dir_yaml, "w") as file:
    #     yaml.dump(sorted_paths, file, default_flow_style=False, sort_keys=False)


# load the optimized params and run simulation in two environments
# load the paths from the yaml file
with open(dir_yaml) as file:
    paths = yaml.load(file, Loader=yaml.FullLoader)
sim_dir = os.path.join(base_dir, "sim_results")
if not os.path.exists(sim_dir):
    os.makedirs(sim_dir)

# rng = jax.random.PRNGKey(traj_opt.seed)
# rng = jax.random.split(rng, 1)


sim = False
test_num_env = 500
num_envs.insert(0, 0)
print(num_envs)
if sim:
    rng = jax.random.PRNGKey(10)
    rng = jax.random.split(rng, test_num_env)
    for num_env in num_envs:
        traj_opt.num_env = num_env
        traj_opt.time_steps = 1000
        config.traj_opt = False
        env = Exo(config)

        optimizer = TrajectoryOptimizer(config, traj_opt, param_keys, param_values, cost_terms, cost_weights)
        optimizer.init_vec_env(rng)

        if num_env == 0:
            optimizer.save_dir = os.path.join(sim_dir, "num_env_" + str(num_env))
            # make the directory if it doesn't exist
            if not os.path.exists(optimizer.save_dir):
                os.makedirs(optimizer.save_dir)
            params = dict(zip(param_keys, param_values))
            vec_states = optimizer.simulate_vec_env(rng, params)
            print("simulate nominal")
        else:
            result_path = paths["num_env" + str(num_env)]
            result_config_path = os.path.join(result_path, "config.yaml")
            with open(result_config_path) as file:
                traj_config = yaml.load(file, Loader=yaml.FullLoader)
            param_date = traj_config["date"]

            opt_config = optimizer.load_config(date=param_date, base_dir="", folder=result_path, best_flag=True)

            optimizer.save_dir = os.path.join(sim_dir, "num_env_" + str(num_env))
            # make the directory if it doesn't exist
            if not os.path.exists(optimizer.save_dir):
                os.makedirs(optimizer.save_dir)
            vec_states = optimizer.simulate_vec_env(rng, opt_config["optimized_params"])
            print("simulate env ", num_env)
        # save vec_states to a pickle file

        with open(os.path.join(optimizer.save_dir, "vec_states.pkl"), "wb") as f:
            print("saving vec_states to", os.path.join(optimizer.save_dir, "vec_states.pkl"))
            pickle.dump(vec_states, f)


@jax.vmap
def getLinVel(data):
    """Get the linear velocity in the x direction."""
    qvel = math.rotate(data.qvel[:3], math.quat_inv(data.qpos[3:7]))
    return qvel[0]


# distinct_colors = [
#     '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
#     '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
#     '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
#     '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
#     '#ffffff', '#000000'
# ]

# def plotFunction(time, metric_val, num_envs, y_axis_label,keyword,sim_dir):
#     plt.figure(figsize=(10, 6))

#     # Define a color map, you can choose one from matplotlib color maps
#     # colors = plt.cm.viridis(np.linspace(0, 1, len(num_envs)))
#     colors = distinct_colors[:len(num_envs)]

#     # Plot each line with a label and color from the color map
#     for i, color in zip(range(len(num_envs)), colors):
#         lengend_label = str(num_envs[i]) + " envs" if i != 0 else "nominal"
#         plt.plot(time, metric_val[i], label=lengend_label, color=color)

#     # Set labels, title, and legend
#     plt.xlabel("Simulation Time (s)", fontsize=12)
#     plt.ylabel(y_axis_label, fontsize=12)
#     plt.legend(fontsize=10, loc='best')  # Adjust legend location as needed


#     # Add gridlines
#     plt.grid(True)
#     plt.savefig(os.path.join(sim_dir, keyword + ".png"), bbox_inches="tight")
def plotEndStates(metric_val, num_envs, y_axis_label, keyword, sim_dir):
    """Plot the last state values for each environment."""
    plt.figure(figsize=(10, 6))

    # Assuming metric_val[i][-1] gives the state value at t=1000 for the i-th environment setup
    end_states = [val[-1] for val in metric_val]  # Collect the end state for each environment

    # Define colors for each bar
    colors = plt.cm.viridis(np.linspace(0, 1, len(num_envs)))

    # Create a bar plot
    plt.bar(range(len(num_envs)), end_states, color=colors, tick_label=[str(env) + " envs" for env in num_envs])

    # Set labels and title
    plt.xlabel("Number of Environments", fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)
    # plt.title("State Values at t=1000", fontsize=14)

    # Add value labels on top of each bar
    for i, v in enumerate(end_states):
        plt.text(i, v + max(end_states) * 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Ensure the simulation directory exists
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)

    # Save the plot to the specified directory
    plt.savefig(os.path.join(sim_dir, keyword + "_end_states.png"), bbox_inches="tight")
    plt.close()  # Close the figure


def plotFunction(time, metric_val, num_envs, y_axis_label, keyword, sim_dir):
    """Plot the specified metric for each environment."""
    plt.figure(figsize=(10, 6))

    # Define a list of distinct colors and line styles
    distinct_colors = [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
    ]
    line_styles = ["-", "--", "-.", ":"]

    # Create a combination of colors and line styles
    color_and_style = [(color, style) for style in line_styles for color in distinct_colors]

    # Plot each line with a unique color and line style
    for i, (color, style) in zip(range(len(num_envs)), color_and_style):
        legend_label = str(num_envs[i]) + " envs" if i != 0 else "nominal"
        plt.plot(time, metric_val[i], label=legend_label, color=color, linestyle=style, linewidth=2)

    # Set labels, title, and legend
    plt.xlabel("Simulation Time (s)", fontsize=12)
    plt.ylabel(y_axis_label, fontsize=12)
    plt.legend(fontsize=10, loc="best")  # Adjust legend location as needed

    # Add gridlines
    plt.grid(True)

    # Ensure the simulation directory exists
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)

    # Save the plot to the specified directory
    plt.savefig(os.path.join(sim_dir, keyword + ".png"), bbox_inches="tight")
    plt.close()  # Close the figure


plot = False
if plot:
    i = 0
    num_violations = []
    x_pos = []
    z_pos = []
    grf = []
    for num_env in num_envs:
        save_dir = os.path.join(sim_dir, "num_env_" + str(num_env))

        # load the vec_states from the pickle file
        with open(os.path.join(save_dir, "vec_states.pkl"), "rb") as f:
            vec_states = pickle.load(f)

        # check number of violations at each time point
        num_violations.append([float(sum(state.done)) for state in vec_states])

        # check qpos < 0.7
        z_pos.append([float(sum(state.pipeline_state.qpos[:, 2] < 0.7)) for state in vec_states])

        # check grf < 10
        grf.append([float(sum(sum(state.pipeline_state.efc_force[:, env.efc_address]) < 100)) for state in vec_states])

        # check the distance traveled
        x_pos.append([float(jnp.mean(state.pipeline_state.qpos[:, 0])) for state in vec_states])

        # math.rotate(data.qvel[:3], math.quat_inv(data.qpos[3:7]))

        track_lin_vel = [float(jnp.mean(getLinVel(state.pipeline_state))) for state in vec_states]

    # # plot z_pos vs num_envs for each num_env
    # time = [float(env.dt) * i for i in range(1000)]

    # save the data to a pickle file
    with open(os.path.join(sim_dir, "sum_states.pkl"), "wb") as f:
        # save z_pos, grf, x_pos, num_violations, track_lin_vel
        pickle.dump(z_pos, f)
        pickle.dump(grf, f)
        pickle.dump(x_pos, f)
        pickle.dump(num_violations, f)
        pickle.dump(track_lin_vel, f)


# plot z_pos vs num_envs for each num_env
time = [float(env.dt) * i for i in range(1000)]

# load the save data for plotting
with open(os.path.join(sim_dir, "sum_states.pkl"), "rb") as f:
    z_pos = pickle.load(f)
    grf = pickle.load(f)
    x_pos = pickle.load(f)
    num_violations = pickle.load(f)
    track_lin_vel = pickle.load(f)

plotFunction(time, num_violations, num_envs, "num violations", "num_violations", sim_dir)
plotFunction(time, z_pos, num_envs, "num env falling", "z_pos", sim_dir)
plotFunction(time, grf, num_envs, "num env falling", "grf", sim_dir)
plotFunction(time, x_pos, num_envs, "x direction displacement", "x_pos", sim_dir)

plotEndStates(num_violations, num_envs, "num violations", "num_violations", sim_dir)
plotEndStates(z_pos, num_envs, "num env falling", "z_pos", sim_dir)
plotEndStates(grf, num_envs, "num env falling", "grf", sim_dir)
plotEndStates(x_pos, num_envs, "x direction displacement", "x_pos", sim_dir)
# optimizer.simulate(
#     env, opt_config["optimized_params"], num_steps=1000, output_video="mjx_traj_opt_default_" + param_date + ".mp4"
# )

# # get the second environment
# rand_sim_sys, _ = randomizeSlopeGain(
# env.sys, 0, rng, max_angle_degrees=traj_opt.max_angle_degrees, max_gain=traj_opt.max_gain
# )

# rand_sim_sys = rand_sim_sys.tree_replace(
# {
#     "geom_quat": jnp.squeeze(rand_sim_sys.geom_quat, axis=0),
#     "geom_friction": jnp.squeeze(rand_sim_sys.geom_friction, axis=0),
#     "actuator_gainprm": jnp.squeeze(rand_sim_sys.actuator_gainprm, axis=0),
#     "actuator_biasprm": jnp.squeeze(rand_sim_sys.actuator_biasprm, axis=0),
# }
# )

# env.unwrapped.sys = rand_sim_sys
# optimizer.simulate(
# env, opt_config["optimized_params"], num_steps=1000, output_video="mjx_traj_opt_rand_env_" + param_date + ".mp4"
# )
