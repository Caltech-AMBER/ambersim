import hashlib
import os
import pickle
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jp
import mediapy as media
import yaml
from brax.envs.wrappers.training import DomainRandomizationVmapWrapper
from jax import grad, jacfwd, jit
from jax.example_libraries import optimizers

import wandb
from ambersim import ROOT
from ambersim.envs.exo_base import BehavState, Exo, ExoConfig
from ambersim.envs.exo_parallel import (
    CustomVecEnv,
    rand_friction,
    randomizeBoxTerrain,
    randomizeCoMOffset,
    randomizeSlope,
    randomizeSlopeGain,
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
        self, config, param_keys, param_values, cost_terms, cost_weights=None, num_env=200, seed=0, time_steps=400
    ):
        """Initialize the trajectory optimizer."""
        self.config = config
        self.env = Exo(config)
        self.ind_jit_env_reset = jit(self.env.reset)
        self.ind_jit_env_step = jit(self.env.step)
        rng = jax.random.PRNGKey(seed)
        self.rng = jax.random.split(rng, num_env)
        self.timesteps = time_steps
        self.cost_weights = cost_weights if cost_weights is not None else {term: 1.0 for term in cost_terms}

        self.cost_terms = cost_terms

        # Initialize the parameters to optimize
        self.params = dict(zip(param_keys, param_values))

    def init_vec_env(self):
        """Initialize the vectorized environment."""
        # self.randomization_fn=partial(rand_friction, rng=self.rng)
        self.randomize_func = partial(randomizeSlopeGain, rng=self.rng, plane_ind=0, max_angle_degrees=5)
        self.domain_env = CustomVecEnv(self.env, randomization_fn=self.randomize_func)
        self.jit_env_reset = jit(self.domain_env.reset)
        self.jit_env_step = jit(self.domain_env.step)

    def updateParams(self, state, params):
        """Update the parameters in the state."""
        for key in params.keys():
            state.info[key] = state.info[key].at[:].set(params[key])
            # jax.debug.print("state.info[key]: {}", state.info[key])
        return state

    def process_metrics(self, metric, flip_sign=False):
        """Process the metric to remove inf and nan values."""
        metric = jp.array(metric)
        if flip_sign:
            metric = -metric
        metric = jp.where(jp.isnan(metric), 1000, metric)
        metric = jp.where(jp.isinf(metric), 1000, metric)
        return metric

    def eval_cost(self, params):
        """Evaluate the cost function."""
        costs_dict = {term: [] for term in self.cost_terms}
        if "alpha" in params and "alpha_base" in params:
            q_init, dq_init = self.env.getBezInitialConfig(
                params["alpha"], params["alpha_base"], self.env.step_dur[BehavState.Walking]
            )

        elif "alpha" in params:
            q_init, dq_init = self.env.getBezInitialConfig(
                params["alpha"], self.env.alpha_base, self.env.step_dur[BehavState.Walking]
            )

        state = self.jit_env_reset(self.rng, q_init, dq_init, BehavState.Walking)
        state = self.updateParams(state, params)
        init_pos_x = state.pipeline_state.qpos[:, 0]
        t_values = []
        for _ in range(self.timesteps):
            state = self.jit_env_step(state, jp.zeros((self.rng.shape[0], self.env.action_size)))
            t_values.append(state.pipeline_state.time)

            for term in self.cost_terms:
                if term != "survival":
                    costs_dict[term].append(-state.info["reward_tuple"][term])
                else:
                    costs_dict[term].append(sum(state.done))

        # Post-process specific metrics
        step_length = state.pipeline_state.qpos[:, 0] - init_pos_x

        if "mechanical_power" in self.cost_terms:
            mcot = env.mcot(jp.array(t_values).T, step_length, jp.array(costs_dict["mechanical_power"]).T)
            costs_dict["mechanical_power"] = self.process_metrics(mcot)

        # Aggregate and process costs
        total_cost = 0.0

        for term in self.cost_terms:
            term_cost = jp.sum(jp.array(costs_dict[term])) * self.cost_weights.get(term, 1.0)
            if term != "mechanical_power":
                term_cost = self.process_metrics(term_cost)
            total_cost += term_cost

        return total_cost / self.rng.shape[0]

    def train(self, num_steps=5, opt_step_size=1e-1, file_name="optimized_params.pkl"):
        """Run the optimization loop."""
        self.init_vec_env()
        self.logger = LoggerFactory.get_logger("wandb", self.save_dir)

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size=opt_step_size)
        opt_state = self.opt_init(self.params)
        wandb.init(project="traj_opt", config={"num_steps": num_steps, "step_size": opt_step_size})
        for i in range(num_steps):
            print("Optimized parameters: ", self.get_params(opt_state))
            value, opt_state = self.optimization_step(i, opt_state)
            wandb.log({"step": i, "value": float(jax.device_get(value))})

        full_file = os.path.join(self.save_dir, file_name)
        print(f"Saving optimized params to: {full_file}")
        with open(full_file, "wb") as file:
            pickle.dump(self.get_params(opt_state), file)

    def optimization_step(self, step_num, opt_state):
        """Perform one optimization step."""
        grads = jacfwd(self.eval_cost)(self.get_params(opt_state))
        jax.debug.print("grads: {}", grads)
        opt_state = self.opt_update(step_num, grads, opt_state)
        return self.eval_cost(self.get_params(opt_state)), opt_state

    def simulate(self, env, optimized_params, num_steps=200, output_video="sim_video.mp4"):
        """Simulate the trajectory using the optimized parameters."""
        if "alpha" in optimized_params and "alpha_base" in optimized_params:
            q_init, dq_init = env.getBezInitialConfig(
                optimized_params["alpha"], optimized_params["alpha_base"], self.env.step_dur[BehavState.Walking]
            )

        elif "alpha" in optimized_params:
            q_init, dq_init = env.getBezInitialConfig(
                optimized_params["alpha"], self.env.alpha_base, self.env.step_dur[BehavState.Walking]
            )

        jit_env_reset = jit(env.reset)
        jit_env_step = jit(env.step)
        state = jit_env_reset(jax.random.PRNGKey(1), q_init, dq_init, BehavState.Walking)
        state = self.updateParams(state, optimized_params)
        images = []
        env.getRender()
        for _ in range(num_steps):
            state = jit_env_step(state, jp.zeros(self.env.action_size))
            images.append(env.get_image(state.pipeline_state))
        media.write_video(os.path.join(self.save_dir, output_video), images, fps=1.0 / self.env.dt)

    def run_base_sim(self, rng, alpha, num_steps=400, output_video="nominal_policy_video.mp4"):
        """Run the simulation basic version."""
        self.env.getRender()

        state = self.ind_jit_env_reset(rng, BehavState.Walking)
        state.info["alpha"] = alpha
        images = []

        for _ in range(num_steps):
            state = self.ind_jit_env_step(state, jp.zeros(self.env.action_size))
            images.append(self.env.get_image(state.pipeline_state))
        media.write_video(output_video, images, fps=1.0 / self.env.dt)
        return

    def run_sim_from_standing(self, rng, num_steps=400, output_video="nominal_policy_from_standing_video.mp4"):
        """Run the simulation from standing position."""
        # start from standing, wait for 0.5 second, go to startingpos, wait for 0.5 second, start walking, walk 2 steps, and then go to stopping pos
        self.env.getRender()
        # currentState = BehavState.ToLoading
        state = self.ind_jit_env_reset(rng, BehavState.WantToStart)
        images = []

        maxErrorTrigger = 0.01
        minTransitionTime = 0.05

        log_items = ["qpos", "qvel", "qfrc_actuator"]
        logged_data_per_step = []

        prev_domain = state.info["domain_info"]["domain_idx"]

        rollouts = []

        for _ in range(num_steps):
            state = self.env.ind_jit_env_step(state, jp.zeros(12))  # Replace with your control strategy

            # Log the time taken for each step
            logged_data = {}
            logged_data = self.log_data(state.pipeline_state, log_items, logged_data)
            logged_data = self.log_state_info(
                state.info, ["state", "domain_info", "tracking_err", "joint_desire"], logged_data
            )
            logged_data_per_step.append(logged_data)
            # Log the time taken for each step

            state_change = False
            # #check state transition
            if state.info["state"] == BehavState.ToLoading:
                jax.debug.print("state: ToLoading")
                if self.state_condition_met(maxErrorTrigger, minTransitionTime, state.pipeline_state, state.info):
                    state.info["state"] = BehavState.Loading

                    state_change = True

            elif state.info["state"] == BehavState.Loading:
                jax.debug.print("state: Loading")
                if self.state_condition_met(maxErrorTrigger, minTransitionTime, state.pipeline_state, state.info):
                    state.info["state"] = BehavState.WantToStart
                    state_change = True

            elif state.info["state"] == BehavState.WantToStart:
                jax.debug.print("state: WantToStart")
                jax.debug.print("blended_action: {}", state.info["blended_action"])
                jax.debug.print("joint_desire: {}", state.info["joint_desire"])
                jax.debug.print("current joint config: {}", state.pipeline_state.qpos[-self.model.nu :])
                minTransitionTime = 2 * self.step_dur[BehavState.WantToStart]
                if self.state_condition_met(maxErrorTrigger, minTransitionTime, state.pipeline_state, state.info):
                    state.info["state"] = BehavState.Walking
                    jax.debug.print("tracking_err: {}", state.info["tracking_err"])
                    jax.debug.print("starting pos: {}", self._q_default[BehavState.WantToStart.value, -self.model.nu :])
                    state_change = True

            elif state.info["state"] == BehavState.Walking:
                jax.debug.print("state: Walking")
                jax.debug.print("blended_action: {}", state.info["blended_action"])
                jax.debug.print("joint_desire: {}", state.info["joint_desire"])
                if state.info["domain_info"]["domain_idx"] != prev_domain:
                    state.info["offset"] = state.pipeline_state.qpos[-self.model.nu :] - self.getNominalDesire(state)[0]
                    jax.debug.print("offset: {}", state.info["offset"])
                    prev_domain = state.info["domain_info"]["domain_idx"]
                    jax.debug.print("domain changed to {}", state.info["domain_info"]["domain_idx"])
                    # state_change = True

            if state_change:
                state.info["domain_info"]["step_start"] = state.pipeline_state.time
                state.info["offset"] = state.pipeline_state.qpos[-self.model.nu :] - self.getNominalDesire(state)[0]

                jax.debug.print("offset: {}", state.info["offset"])

            rollouts.append(state)
            images.append(self.env.get_image(state.pipeline_state))

        media.write_video(output_video, images, fps=1.0 / self.dt)

        log_file = "logged_data.json"
        rollout_file = "rollout_data.pkl"
        with open(log_file, "wb") as f:
            pickle.dump(logged_data_per_step, f)

        with open(rollout_file, "wb") as f:
            pickle.dump(rollouts, f)
        self.plot_logged_data(logged_data_per_step)

        return

    def generate_hash(self, date=None):
        """Generate a unique hash based on the param_keys, cost_terms, and a given date."""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        hash_input = str(self.params.keys()) + str(self.cost_terms) + date
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def save_config(self, base_dir="configurations"):
        """Save the current configuration parameters to a file using a generated hash as the filename."""
        hash_path = self.generate_hash()
        full_path = os.path.join(base_dir, hash_path)
        self.save_dir = full_path

        config_data = {
            "params": list(self.params.keys()),  # Convert dict_keys to a list
            "cost_terms": self.cost_terms,
            "cost_weights": self.cost_weights,
            "date": datetime.now().strftime("%Y%m%d"),
        }
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(full_path, exist_ok=True)

        full_config_path = os.path.join(full_path, "config.yaml")
        with open(full_config_path, "w") as yaml_file:
            yaml.dump(config_data, yaml_file, default_flow_style=None)

        return full_path

    def load_config(self, date, base_dir="configurations", file_name="optimized_params.pkl", export_yaml=True):
        """Load configuration parameters from a file identified by the hash generated with a specific date."""
        hash_path = self.generate_hash(date=date)
        full_path = os.path.join(base_dir, hash_path)
        full_config_path = os.path.join(full_path, "config.yaml")
        with open(full_config_path, "r") as file:
            config = yaml.safe_load(file)
        self.save_dir = full_path
        file = os.path.join(self.save_dir, file_name)
        print(f"Loading config from: {file}")
        # config = {}
        with open(file, "rb") as optimized_File:
            config["optimized_params"] = pickle.load(optimized_File)
        print(f"optimized_params: {config['optimized_params']}")

        if os.path.exists(file_name):
            print(f"File:{file_name} exists")

        if export_yaml:
            gait_params_file = os.path.join(ROOT, "..", "models", "exo", env.config.jt_traj_file)
            with open(gait_params_file, "r") as file:
                gait_params = yaml.safe_load(file)
            gait_params["coeff_jt"] = config["optimized_params"]["alpha"].T.ravel().astype(float).tolist()
            output_file = os.path.join(full_path, "optimized_params.yaml")
            with open(output_file, "w") as file:
                yaml.dump(gait_params, file, default_flow_style=None)

        return config

    def export_yaml(self, filename):
        """Export the optimized parameters to a yaml file."""
        config = {}
        with open(filename, "rb") as optimized_File:
            config["optimized_params"] = pickle.load(optimized_File)
        gait_params_file = os.path.join(ROOT, "..", "models", "exo", env.config.jt_traj_file)
        with open(gait_params_file, "r") as file:
            gait_params = yaml.safe_load(file)
        gait_params["coeff_jt"] = config["optimized_params"]["alpha"].T.ravel().astype(float).tolist()
        output_file = os.path.join("multi_merged_optimized_params.yaml")
        with open(output_file, "w") as file:
            yaml.dump(gait_params, file, default_flow_style=None)


if __name__ == "__main__":
    # Example usage:
    config = ExoConfig()
    config.slope = True
    config.impact_based_switching = False
    config.jt_traj_file = "merged_multicontact.yaml"
    config.no_noise = False
    config.hip_regulation = False
    env = Exo(config)
    param_keys = ["alpha"]
    # with open("multi_no_pos_optimized_params_update.pkl", "rb") as file:
    #     init_guess = pickle.load(file)
    # param_values = [init_guess["alpha"]]  # Example initial conditions
    param_values = [env.alpha]  # Example initial conditions
    cost_terms = [
        "base_smoothness_reward",
        "jt_smoothness_reward",
        "tracking_lin_vel_reward",
        "mechanical_power",
        "survival",
    ]
    # param_keys = ['hip_regulator_gain']
    # param_values = [config.hip_regulator_gain]  # Example initial conditions
    # param_keys = ['alpha','hip_regulator_gain']
    # param_values = [env.alpha,config.hip_regulator_gain]  # Example initial conditions
    cost_weights = {
        "base_smoothness_reward": 1.0,
        "jt_smoothness_reward": 1.0,
        "tracking_lin_vel_reward": 1.0,
        "mechanical_power": 1.0,
        "survival": 100.0,
    }
    optimizer = TrajectoryOptimizer(config, param_keys, param_values, cost_terms, cost_weights, time_steps=300)
    # filename = "multi_no_pos_optimized_params_update_v2.pkl" #either this or without v2 is the old result

    # optimizer.run_base_sim(jax.random.PRNGKey(0),init_guess["alpha"],output_video="mjx_version.mp4")
    # breakpoint()

    # filename = "/home/kli5/ambersim/ambersim/envs/multi_no_pos_optimized_params_update.pkl"
    # optimizer.export_yaml(filename)

    # To load a config with a specific old date
    # old_date = '20230101'  # example old date
    # loaded_config = TrajectoryOptimizer.load_config("configurations", param_keys, cost_terms, old_date)
    # print(f"Loaded config with old date: {loaded_config}")

    # # To load the config later
    # loaded_config = optimizer.load_config(path)
    # print(f"Loaded config: {loaded_config}")

    train = True
    # train = False
    if train:
        path = optimizer.save_config()
        print(f"Config saved at: {path}")
        optimizer.train(num_steps=15, opt_step_size=1e-4)
    else:
        config = ExoConfig()
        config.slope = False
        config.impact_based_switching = False
        config.jt_traj_file = "merged_multicontact.yaml"
        config.no_noise = True
        config.hip_regulation = False
        env = Exo(config)

        param_date = "20240102"
        opt_config = optimizer.load_config(date=param_date)
        # nominal = {"alpha": optimizer.env.alpha}
        optimizer.simulate(
            env, opt_config["optimized_params"], num_steps=1000, output_video="multi_traj_opt_" + param_date + ".mp4"
        )
        # optimizer.simulate(nominal, num_steps=1000, output_video="nominal.mp4")
