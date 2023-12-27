import os
import pickle
from functools import partial

import jax
import jax.numpy as jp
import mediapy as media
from brax.envs.wrappers.training import DomainRandomizationVmapWrapper
from jax import grad, jacfwd, jit
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
from ambersim.logger.logger import LoggerFactory

# Set environment variables outside the class as they are global settings
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MUJOCO_GL"] = "egl"


class TrajectoryOptimizer:
    """Class to optimize trajectory parameters."""

    def __init__(
        self, config, param_keys, param_values, cost_terms, log_dir="exo_results", num_env=20, seed=0, time_steps=220
    ):
        """Initialize the trajectory optimizer."""
        self.config = config
        self.env = Exo(config)
        self.ind_jit_env_reset = jit(self.env.reset)
        self.ind_jit_env_step = jit(self.env.step)
        rng = jax.random.PRNGKey(seed)
        self.rng = jax.random.split(rng, num_env)
        self.timesteps = time_steps
        self.logger = LoggerFactory.get_logger("wandb", log_dir)

        self.cost_terms = cost_terms

        # Initialize the parameters to optimize
        self.params = dict(zip(param_keys, param_values))

    def init_vec_env(self):
        """Initialize the vectorized environment."""
        # self.randomization_fn=partial(rand_friction, rng=self.rng)
        self.randomize_func = partial(randomizeSlope, rng=self.rng, plane_ind=0, max_angle_degrees=3)
        self.domain_env = CustomVecEnv(self.env, randomization_fn=self.randomize_func)
        self.jit_env_reset = jit(self.domain_env.reset)
        self.jit_env_step = jit(self.domain_env.step)

    def updateParams(self, state, params):
        """Update the parameters in the state."""
        for key in params.keys():
            state.info[key] = state.info[key].at[:].set(params[key])
            # jax.debug.print("state.info[key]: {}", state.info[key])
        return state

    def getCurrentCost(self, state):
        """Get the current cost of the state."""
        # TODO: need to update this to use the cost terms
        current_cost = 0.0
        for term_name in self.cost_terms:
            reward_value = state.info["reward_tuple"].get(term_name, 0.0)
            current_cost -= self.process_metrics(reward_value)
        return current_cost

    def process_metrics(self, metric, flip_sign=False):
        """Process the metric to remove inf and nan values."""
        metric = jp.array(metric)
        if flip_sign:
            metric = -metric
        metric = jp.where(jp.isnan(metric), 1000, metric)
        metric = jp.where(jp.isinf(metric), 1000, metric)
        return metric

    def eval_cost(self, params):
        """Evaluate the cost of the trajectory."""
        costs = 0.0
        # step_dur = 1.0
        penalty = 1000.0
        survive_reward = 5.0
        P_values = []
        t_values = []
        state = self.jit_env_reset(self.rng, BehavState.Walking)
        state = self.updateParams(state, params)
        # state.info["alpha"] = state.info["alpha"].at[:].set(alpha)
        # init_foot_pos = state.pipeline_state.geom_xpos[:, env.foot_geom_idx[0], 0]
        init_pos_x = state.pipeline_state.qpos[:, 0]

        track_err = []
        terminate_status = []
        for _ in range(self.timesteps):
            state = self.jit_env_step(state, jp.zeros((self.rng.shape[0], self.env.action_size)))

            P_values.append(state.info["mechanical_power"])
            t_values.append(state.pipeline_state.time)

            if state.info["state"][0] == BehavState.WantToStart:
                state.info["state"] = state.info["state"].at[:].set(BehavState.Walking)

            if sum(state.done) and state.pipeline_state.time[0] > 0.02:
                terminate_status.append(sum(state.done))
                costs = costs + penalty * sum(state.done)

            track_err.append(state.info["reward_tuple"]["tracking_pos_reward"])
            costs = costs - survive_reward * (state.done.shape[0] - sum(state.done))

            # step_length = state.pipeline_state.geom_xpos[:, env.foot_geom_idx[0], 0] - init_foot_pos
        step_length = state.pipeline_state.qpos[:, 0] - init_pos_x
        # jax.debug.breakpoint()
        # jax.debug.print("step_length: {}", step_length)
        # jax.debug.print("track_err: {}", track_err)
        mcot = env.mcot(jp.array(t_values).T, step_length, jp.array(P_values).T)
        # set where there's inf or nan mcot to 1000
        mcot = self.process_metrics(mcot)
        # track_err = self.process_metrics(track_err, flip_sign=True)
        costs = costs + 100 * mcot  # + sum(jp.array(track_err))
        jax.debug.print("mcot: {}", env.mcot(jp.array(t_values).T, step_length, jp.array(P_values).T))
        jax.debug.print("costs: {}", costs)
        jax.debug.print("sum(state.done): {}", jp.array(terminate_status))

        return jp.sum(jp.array(costs)) / (self.rng.shape[0])

    def train(self, num_steps=5, opt_step_size=1e-1, file_name="optimized_params.pkl"):
        """Run the optimization loop."""
        self.init_vec_env()
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size=opt_step_size)
        opt_state = self.opt_init(self.params)
        wandb.init(project="traj_opt", config={"num_steps": num_steps, "step_size": opt_step_size})
        for i in range(num_steps):
            print("Optimized parameters: ", self.get_params(opt_state))
            value, opt_state = self.optimization_step(i, opt_state)
            wandb.log({"step": i, "value": value})

        with open(file_name, "wb") as file:
            pickle.dump(self.get_params(opt_state), file)

    def optimization_step(self, step_num, opt_state):
        """Perform one optimization step."""
        grads = jacfwd(self.eval_cost)(self.get_params(opt_state))
        jax.debug.print("grads: {}", grads)
        opt_state = self.opt_update(step_num, grads, opt_state)
        return self.eval_cost(self.get_params(opt_state)), opt_state

    def simulate(self, params_file, num_steps=200, output_video="sim_video.mp4"):
        """Simulate the trajectory using the optimized parameters."""
        with open(params_file, "rb") as file:
            optimized_params = pickle.load(file)
        state = self.ind_jit_env_reset(jax.random.PRNGKey(1), BehavState.Walking)
        state = self.updateParams(state, optimized_params)
        images = []
        self.env.getRender()
        for _ in range(num_steps):
            state = self.ind_jit_env_step(state, jp.zeros(self.env.action_size))
            images.append(self.env.get_image(state.pipeline_state))
        media.write_video(os.path.join(self.logger.log_dir, output_video), images, fps=1.0 / self.env.dt)


if __name__ == "__main__":
    # Example usage:
    config = ExoConfig()
    config.slope = False
    config.impact_based_switching = False
    config.jt_traj_file = "merged_multicontact.yaml"
    config.no_noise = False
    config.hip_regulation = False
    env = Exo(config)
    param_keys = ["alpha"]
    with open("multi_no_pos_optimized_params_update.pkl", "rb") as file:
        init_guess = pickle.load(file)
    param_values = [init_guess["alpha"]]  # Example initial conditions
    cost_terms = ["tracking_pos_reward", "tracking_vel_reward"]
    # param_keys = ['hip_regulator_gain']
    # param_values = [config.hip_regulator_gain]  # Example initial conditions
    # param_keys = ['alpha','hip_regulator_gain']
    # param_values = [env.alpha,config.hip_regulator_gain]  # Example initial conditions
    optimizer = TrajectoryOptimizer(config, param_keys, param_values, cost_terms, time_steps=120)
    filename = "multi_no_pos_optimized_params_update_v2.pkl"

    # train = True
    train = False
    if train:
        optimizer.train(num_steps=30, opt_step_size=1e-7, file_name=filename)
    else:
        config = ExoConfig()
        config.slope = False
        config.impact_based_switching = False
        config.jt_traj_file = "merged_multicontact.yaml"
        config.no_noise = True
        config.hip_regulation = False
        env = Exo(config)

        optimizer.simulate(filename, num_steps=1000, output_video="multi_no_pos_traj_opt_update_v2.mp4")
