import os
import pickle
from itertools import product
from typing import List

import jax
import jax.numpy as jnp
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
config.impact_threshold = 400.0
config.jt_traj_file = "default_bez.yaml"
config.physics_steps_per_control_step = 5
config.reset_noise_scale = 1e-3
config.no_noise = False
config.controller.hip_regulation = False
env = Exo(config)
param_keys = ["alpha"]
param_values = [env.alpha]  # Example initial conditions

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
    "tracking_err": 10.0,
    "tracking_pos_reward": 1.0,
    "tracking_lin_vel_reward": 10.0,
    "tracking_ang_vel_reward": 1.0,
    "survival": 100.0,
}

traj_opt = TrajOptConfig()
traj_opt.num_steps = 50

# loop through different num env values;
num_envs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
paths = {}
dir_yaml = "traj_results/num_envs_dir.yaml"

train = False
if train:
    for num_env in num_envs:
        traj_opt.num_env = num_env

        optimizer = TrajectoryOptimizer(config, traj_opt, param_keys, param_values, cost_terms, cost_weights)

        path = optimizer.save_config(base_dir="traj_results")
        print(f"Config saved at: {path}")
        optimizer.train()

        paths[str(num_env)] = path

        # save the paths to a yaml file
        with open(dir_yaml, "w") as file:
            documents = yaml.dump(paths, file)

    # Sort the paths dictionary by keys (convert keys to integers for sorting)
    sorted_paths = {k: paths[k] for k in sorted(paths, key=lambda x: int(x))}

    # Save the sorted paths to a yaml file with pretty formatting
    with open(dir_yaml, "w") as file:
        yaml.dump(sorted_paths, file, default_flow_style=False, sort_keys=False)


# load the optimized params and run simulation in two environments
# load the paths from the yaml file
with open(dir_yaml) as file:
    paths = yaml.load(file, Loader=yaml.FullLoader)
sim_dir = "traj_results/sim_results_rand"
if not os.path.exists(sim_dir):
    os.makedirs(sim_dir)

# rng = jax.random.PRNGKey(traj_opt.seed)
# rng = jax.random.split(rng, 1)


sim = False
if sim:
    test_num_env = 500
    rng = jax.random.PRNGKey(10)
    rng = jax.random.split(rng, test_num_env)
    for num_env in num_envs:
        result_path = paths[str(num_env)]
        result_config_path = os.path.join(result_path, "config.yaml")
        with open(result_config_path) as file:
            traj_config = yaml.load(file, Loader=yaml.FullLoader)
        param_date = traj_config["date"]

        traj_opt.num_env = num_env
        traj_opt.time_steps = 1000
        # load the optimized params and run simulation in two environments
        # the first environment is the original environment

        config = ExoConfig()
        config.slope = True
        config.impact_based_switching = True
        config.impact_threshold = 400.0
        config.jt_traj_file = "default_bez.yaml"
        config.physics_steps_per_control_step = 5
        config.reset_noise_scale = 1e-3
        config.no_noise = False
        config.controller.hip_regulation = False
        env = Exo(config)

        optimizer = TrajectoryOptimizer(config, traj_opt, param_keys, param_values, cost_terms, cost_weights)
        optimizer.init_vec_env(rng)

        opt_config = optimizer.load_config(date=param_date, base_dir="", folder=result_path, best_flag=True)

        nominal = {"alpha": optimizer.env.alpha}
        optimizer.save_dir = os.path.join(sim_dir, "num_env_" + str(num_env))
        # make the directory if it doesn't exist
        if not os.path.exists(optimizer.save_dir):
            os.makedirs(optimizer.save_dir)

        vec_states = optimizer.simulate_vec_env(rng, opt_config["optimized_params"])

        # save vec_states to a pickle file

        with open(os.path.join(optimizer.save_dir, "vec_states.pkl"), "wb") as f:
            pickle.dump(vec_states, f)


@jax.vmap
def getLinVel(data):
    """Get the linear velocity in the x direction."""
    qvel = math.rotate(data.qvel[:3], math.quat_inv(data.qpos[3:7]))
    return qvel[0]


plot = True
if plot:
    import jax.numpy as jp
    import matplotlib.pyplot as plt

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
        x_pos.append([float(jp.mean(state.pipeline_state.qpos[:, 0])) for state in vec_states])

        # math.rotate(data.qvel[:3], math.quat_inv(data.qpos[3:7]))

        track_lin_vel = [float(jp.mean(getLinVel(state.pipeline_state))) for state in vec_states]

    # plot z_pos vs num_envs for each num_env
    time = [float(env.dt) * i for i in range(1000)]
    plt.figure()
    for i in range(len(num_envs)):
        plt.plot(time, num_violations[i], label=str(num_envs[i]))
        plt.xlabel("simulation time (s)")
        plt.ylabel("num violations")
        plt.legend()
        plt.savefig(os.path.join(sim_dir, "num_violations.png"), bbox_inches="tight")

    plt.figure()
    for i in range(len(num_envs)):
        plt.plot(time, z_pos[i], label=str(num_envs[i]))
        plt.xlabel("simulation time (s)")
        plt.ylabel("num env failling")
        plt.legend()
        plt.savefig(os.path.join(sim_dir, "z_pos.png"), bbox_inches="tight")

    plt.figure()
    for i in range(len(num_envs)):
        plt.plot(time, grf[i], label=str(num_envs[i]))
        plt.xlabel("simulation time (s)")
        plt.ylabel("num env failling")
        plt.legend()
        plt.savefig(os.path.join(sim_dir, "grf.png"), bbox_inches="tight")

    plt.figure()
    for i in range(len(num_envs)):
        plt.plot(time, x_pos[i], label=str(num_envs[i]))
        plt.xlabel("simulation time (s)")
        plt.ylabel("x direction displacement")
        plt.legend()
        plt.savefig(os.path.join(sim_dir, "x_pos.png"), bbox_inches="tight")

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
