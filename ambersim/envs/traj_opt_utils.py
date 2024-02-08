from dataclasses import dataclass, field

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

from ambersim.envs.exo_base import BehavState, Exo, ExoConfig


@dataclass
class TrajOptConfig:
    """Data class for storing default values for Trajectory Optimizer configuration."""

    # Define default values for each parameter
    num_env: int = 10
    num_init: int = 100
    seed: int = 0
    time_steps: int = 2
    num_steps: int = 1
    opt_step_size: float = 1e-3
    file_name: str = "optimized_params.pkl"
    output_video: str = "sim_video.mp4"
    simulation_steps: int = 200
    max_angle_degrees: float = 2.0
    max_gain: float = 500.0
    geom_indices: jp.ndarray = field(default_factory=lambda: jp.arange(1))
    barrier: bool = True

    def to_dict(self):
        """Convert the dataclass to a dictionary."""
        config_dict = self.__dict__.copy()
        config_dict["geom_indices"] = config_dict["geom_indices"].tolist()

        return config_dict


def h(x, x_star, r):
    """Barrier function condition for the discrete-time Lyapunov function.

    Args:
        x (jp.ndarray): Current state.
        x_star (jp.ndarray): Desired state.
        alpha (float): Barrier function condition.

    Returns:
        float: Barrier function condition.
    """
    return r - jp.linalg.norm(x - x_star, axis=-1) ** 2


def delta_h_hat(xk: jax.Array, xk_plus_1: jax.Array, xstar: jax.Array, r: float):
    """Calculate delta h condition."""
    # delta_h_hat = 1/N * \sum [h(xk_plus_1) - h(xk)]
    # h_k = jax.vmap(h,in_axes=(0,None,None))(xk,xstar,r)
    # h_k_plus_1 = jax.vmap(h,in_axes=(0,None,None))(xk_plus_1,xstar,r)
    h_k = h(xk, xstar, r)
    h_k_plus_1 = h(xk_plus_1, xstar, r)

    delta_h_over_h = (h_k_plus_1 - h_k) / h_k

    return delta_h_over_h


def get_x_k(env, state):
    """Get the start state from the MuJoCo simulation data."""
    data = state.pipeline_state
    domain_idx = state.info["domain_info"]["domain_idx"]
    com2st = data.subtree_com[:, 1, :] - data.geom_xpos[:, env.foot_geom_idx[domain_idx[0]], 0:3]
    return jp.concatenate([com2st, data.cvel[:, 1, :]], axis=1)
    # return


def get_x_k_plus_1(env, state):
    """Get the end state from the MuJoCo simulation data."""
    # R = env._remap_coeff()
    # x_k_plus_1 = R @data.qvel.T
    # return x_k_plus_1.T
    # x_k_plus_1 = env.R_base[0:3,0:3] @ env.getCoMs2StPos(state)
    data = state.pipeline_state
    com_vel = env.R_base @ data.cvel[:, 1, :].T
    com2st = (
        env.R_base[0:3, 0:3]
        @ (
            data.subtree_com[:, 1, :]
            - data.geom_xpos[:, env.foot_geom_idx[state.info["domain_info"]["domain_idx"][0]], 0:3]
        ).T
    )
    return jp.concatenate([com2st.T, com_vel.T], axis=1)


# evaluate center of mass and also center of pressure instead?
# make the barrier condition more generic so it's easier to test out


def get_x_star(qpos, qvel, rng, total_env):
    """Get the desired state from the nominal states."""
    from mujoco.mjx._src import smooth

    config = ExoConfig()
    config.no_noise = True
    env = Exo()
    states = env.reset(rng, qpos, qvel, BehavState.Walking)
    data = states.pipeline_state
    domain_idx = states.info["domain_info"]["domain_idx"]
    com2st = data.subtree_com[1, :] - data.geom_xpos[env.foot_geom_idx[domain_idx], 0:3]
    com_states = jp.concatenate([com2st, data.cvel[1, :]], axis=0)
    return jp.tile(com_states, (total_env, 1))


if __name__ == "__main__":
    from ambersim.envs.exo_traj_opt import TrajectoryOptimizer

    config = ExoConfig()
    config.rand_terrain = False
    config.traj_opt = True
    config.impact_based_switching = True
    config.impact_threshold = 400.0
    config.jt_traj_file = "default_bez.yaml"
    config.physics_steps_per_control_step = 5
    config.no_noise = True
    config.controller.hip_regulation = False
    config.controller.cop_regulation = False
    env = Exo(config)
    param_keys = ["alpha"]
    param_values = [env.alpha]  # Example initial conditions

    traj_opt_config = TrajOptConfig()
    optimizer = TrajectoryOptimizer(config, traj_opt_config, param_keys, param_values, ["barrier"], {"barrier": 1.0})

    train = False

    if train:
        path = optimizer.save_config()
        print(f"Config saved at: {path}")
        optimizer.train()
    else:
        config.traj_opt = False
        config.no_noise = True
        config.controller.hip_regulation = False
        config.controller.cop_regulation = False
        env = Exo(config)

        param_date = "20240120"

        nominal_flag = False
        opt_config = optimizer.load_config(date=param_date, best_flag=True)
        if nominal_flag:
            nominal = dict(zip(param_keys, param_values))

            optimizer.simulate(
                env,
                nominal,
                num_steps=1000,
                output_video="nominal_" + param_date + ".mp4",
            )

        else:
            # nominal = {"cop_regulator_gain": env.config.controller.cop_regulator_gain}
            # ,"hip_regulator_gain":env.config.controller.hip_regulator_gain,"impact_threshold":env.config.impact_threshold
            optimizer.simulate(
                env,
                opt_config["optimized_params"],
                num_steps=1000,
                output_video="mjx_traj_opt_" + param_date + ".mp4",
            )
