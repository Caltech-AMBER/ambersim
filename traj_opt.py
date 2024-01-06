import os

import jax
import jax.numpy as jnp
import yaml

from ambersim.envs.exo_base import BehavState, Exo, ExoConfig
from ambersim.envs.exo_parallel import randomizeSlopeGain
from ambersim.envs.exo_traj_opt import TrajectoryOptimizer, TrajOptConfig

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

sim_dir = "traj_results/sim_results"


rng = jax.random.PRNGKey(traj_opt.seed)
rng = jax.random.split(rng, 1)
rand_sim_sys, _ = randomizeSlopeGain(
    env.sys, 0, rng, max_angle_degrees=optimizer.config.max_angle_degrees, max_gain=optimizer.config.max_gain
)

rand_sim_sys = rand_sim_sys.tree_replace(
    {
        "geom_quat": jnp.squeeze(rand_sim_sys.geom_quat, axis=0),
        "geom_friction": jnp.squeeze(rand_sim_sys.geom_friction, axis=0),
        "actuator_gainprm": jnp.squeeze(rand_sim_sys.actuator_gainprm, axis=0),
        "actuator_biasprm": jnp.squeeze(rand_sim_sys.actuator_biasprm, axis=0),
    }
)


for num_env in num_envs:
    result_path = paths[str(num_env)]
    traj_opt.num_env = num_env
    optimizer = TrajectoryOptimizer(config, traj_opt, param_keys, param_values, cost_terms, cost_weights)

    # load the optimized params and run simulation in two environments
    # the first environment is the original environment

    config = ExoConfig()
    config.slope = True
    config.impact_based_switching = True
    config.jt_traj_file = "default_bez.yaml"
    config.physics_steps_per_control_step = 5
    config.no_noise = False
    config.controller.hip_regulation = False
    env = Exo(config)

    param_date = "20240105"
    opt_config = optimizer.load_config(date=param_date, best_flag=True)

    nominal = {"alpha": optimizer.env.alpha}
    optimizer.save_dir = os.path.join(sim_dir, "num_env_" + str(num_env))
    # make the directory if it doesn't exist
    if not os.path.exists(optimizer.save_dir):
        os.makedirs(optimizer.save_dir)
    optimizer.simulate(
        env, opt_config["optimized_params"], num_steps=2400, output_video="mjx_traj_opt_default_" + param_date + ".mp4"
    )

    # get the second environment
    env.unwrapped.sys = rand_sim_sys
    optimizer.simulate(
        env, opt_config["optimized_params"], num_steps=240, output_video="mjx_traj_opt_rand_env_" + param_date + ".mp4"
    )
