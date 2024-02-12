import hashlib
import os
import pickle
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jp
import mediapy as media
import yaml
from jax import grad, jacfwd, jit
from jax.example_libraries import optimizers

import wandb
from ambersim import ROOT
from ambersim.envs.exo_base import BehavState, Exo, ExoConfig
from ambersim.envs.exo_parallel import CustomVecEnv, randomizeSlopeGainWeight
from ambersim.envs.exo_tsc import Exo_TSC
from ambersim.envs.traj_opt_utils import TrajOptConfig, delta_h_hat, get_x_k, get_x_k_plus_1, get_x_star, h
from ambersim.logger.logger import LoggerFactory

# # Set environment variables outside the class as they are global settings
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["MUJOCO_GL"] = "egl"


class TrajectoryOptimizer:
    """Class to optimize trajectory parameters."""

    def __init__(self, env_config, traj_opt_config, param_keys, param_values, cost_terms, cost_weights=None):
        """Initialize the trajectory optimizer."""
        self.config = traj_opt_config
        self.env_config = env_config
        self.env = Exo_TSC(env_config)
        self.ind_jit_env_reset = jit(self.env.reset)
        self.ind_jit_env_step = jit(self.env.step)
        rng = jax.random.PRNGKey(self.config.seed)
        self.rng = jax.random.split(rng, self.config.num_env)
        self.timesteps = self.config.time_steps
        self.cost_weights = cost_weights if cost_weights is not None else {term: 1.0 for term in cost_terms}

        self.cost_terms = cost_terms

        # Initialize the parameters to optimize
        self.params = dict(zip(param_keys, param_values))

    def init_vec_env(self, rng=None):
        """Initialize the vectorized environment."""
        if rng is None:
            rng = self.rng

        # self.randomization_fn=partial(rand_friction, rng=self.rng)
        self.randomize_func = partial(
            randomizeSlopeGainWeight,
            torso_idx=self.env.base_frame_idx,
            rng=rng,
            geom_indices=self.config.geom_indices,
            max_angle_degrees=self.config.max_angle_degrees,
            max_gain=self.config.max_gain,
        )
        self.domain_env = CustomVecEnv(self.env, randomization_fn=self.randomize_func)
        self.jit_env_reset = jit(self.domain_env.reset)
        if self.config.barrier:
            self.jit_env_step = jit(self.domain_env.barrier_step)
        else:
            self.jit_env_step = jit(self.domain_env.step)

    def updateParams(self, state, params):
        """Update the parameters in the state."""
        for key in params.keys():
            # check if it's 1d array
            # jax.debug.breakpoint()
            if key == "q_b_init" or key == "dq_b_init":
                continue
            elif len(state.info[key].shape):
                state.info[key] = state.info[key].at[:].set(params[key])
            else:
                state.info[key] = params[key]
            # jax.debug.print("state.info[key]: {}", state.info[key])
        return state

    def process_metrics(self, metric):
        """Process the metric to remove inf and nan values."""
        metric = jp.array(metric)
        metric = jp.where(jp.isnan(metric), 1000, metric)
        metric = jp.where(jp.isinf(metric), 1000, metric)
        return metric

    def simulate_vec_env(self, rng, params):
        """Evaluate the cost function."""
        if "alpha" in params and "alpha_base" in params:
            q_init, dq_init = self.env.getBezInitialConfig(
                params["alpha"], params["alpha_base"], self.env.step_dur[BehavState.Walking]
            )

        elif "alpha" in params:
            q_init, dq_init = self.env.getBezInitialConfig(
                params["alpha"], self.env.alpha_base, self.env.step_dur[BehavState.Walking]
            )
        else:
            q_init = self.env._q_init
            dq_init = self.env._dq_init

        state = self.jit_env_reset(rng, q_init, dq_init, BehavState.Walking)
        state = self.updateParams(state, params)

        states = []
        for _ in range(self.timesteps):
            state = self.jit_env_step(state, jp.zeros((rng.shape[0], self.env.action_size)))
            states.append(state)

        return states

    def eval_barrier_cond(self, params):
        """Evaluate barrier condition."""
        q_init, dq_init = self.obtain_init(params, self.env)

        total_env = self.config.num_env * self.config.num_init
        x_star = get_x_star(q_init, dq_init, jax.random.PRNGKey(0), total_env)

        init_noise = jax.random.uniform(
            jax.random.PRNGKey(1), (self.config.num_init, self.env.model.nv), minval=-0.1, maxval=0.1
        )
        q_init_rand = jp.tile(q_init, (self.config.num_init, 1))
        dq_init_rand = jp.tile(dq_init, (self.config.num_init, 1))
        dq_init_rand = dq_init_rand.at[:, 0 : self.env.model.nv].set(
            dq_init_rand[:, 0 : self.env.model.nv] + init_noise
        )

        q_inits = jp.tile(q_init_rand, (self.config.num_env, 1))
        dq_inits = jp.tile(dq_init_rand, (self.config.num_env, 1))

        state = self.domain_env.barrier_reset(self.rngs_replicated, q_inits, dq_inits, BehavState.Walking)
        state = self.updateParams(state, params)

        x_k = get_x_k(self.env, state)

        r = 0.85
        # check h(xk) > 0
        h_k = h(x_k, x_star, r)

        if jp.any(h_k < 0):
            jax.debug.print("h_k: {}", h_k)
            breakpoint()

        for _ in range(self.timesteps):
            state = self.jit_env_step(state, jp.zeros((total_env, self.env.action_size)))

        # make sure it stop simulate when one step is completed
        # breakpoint()
        x_k_plus_1 = get_x_k_plus_1(self.env, state)
        delta_h_val = delta_h_hat(x_k, x_k_plus_1, x_star, r)
        delta_h_over_h = jp.clip(delta_h_val / h_k, a_max=0)
        barrier_val = jp.sum(-delta_h_over_h)
        jax.debug.print("barrier_val: {}", barrier_val)

        return barrier_val

    def eval_cost(self, params):
        """Evaluate the cost function."""
        costs_dict = {term: [] for term in self.cost_terms}

        q_init, dq_init = self.obtain_init(params, self.env)

        state = self.jit_env_reset(self.rng, q_init, dq_init, BehavState.Walking)
        state = self.updateParams(state, params)
        init_pos_x = state.pipeline_state.qpos[:, 0]
        t_values = []
        for _ in range(self.timesteps):
            state = self.jit_env_step(state, jp.zeros((self.rng.shape[0], self.env.action_size)))
            t_values.append(state.pipeline_state.time)

            for term in self.cost_terms:
                if term == "tracking_err":
                    costs_dict[term].append(sum(state.info["tracking_err"]))
                elif term == "output_err":
                    costs_dict[term].append(sum(state.info["output_err"]))
                elif term == "impact_mismatch":
                    costs_dict[term].append(sum(state.info["domain_info"]["impact_mismatch"]))
                elif term != "survival":
                    costs_dict[term].append(-state.info["reward_tuple"][term])
                else:
                    costs_dict[term].append(sum(state.done))

        # Post-process specific metrics
        step_length = state.pipeline_state.qpos[:, 0] - init_pos_x

        if "mechanical_power" in self.cost_terms:
            mcot = env.mcot(jp.array(t_values).T, step_length, jp.array(costs_dict["mechanical_power"]).T)
            costs_dict["mechanical_power"] = self.env._clip_reward(self.process_metrics(mcot))
            jax.debug.print("mcot: {}", mcot)

        # Aggregate and process costs
        total_cost = 0.0

        for term in self.cost_terms:
            term_cost = jp.sum(jp.array(costs_dict[term])) * self.cost_weights.get(term, 1.0)
            if term != "mechanical_power":
                term_cost = self.process_metrics(term_cost)
            total_cost += term_cost

        jax.debug.breakpoint()
        return total_cost / self.rng.shape[0]

    def optimization_barrier_step(self, step_num, opt_state, clip_value=100.0):
        """Perform one optimization step."""
        grads = jacfwd(self.eval_barrier_cond)(self.get_params(opt_state))
        jax.debug.print("grads: {}", grads)
        # Clip gradients
        clipped_grads = jax.tree_util.tree_map(lambda g: jp.clip(g, -clip_value, clip_value), grads)

        # Debugging: print gradients
        jax.debug.print("Clipped grads: {}", clipped_grads)

        opt_state = self.opt_update(step_num, clipped_grads, opt_state)
        return self.eval_barrier_cond(self.get_params(opt_state)), opt_state

    def train(self, file_name="optimized_params.pkl"):
        """Run the optimization loop."""
        if self.config.barrier:
            self.rngs_replicated = jp.tile(self.rng, (self.config.num_init, 1))
            self.init_vec_env(self.rngs_replicated)
        else:
            self.init_vec_env()
        self.logger = LoggerFactory.get_logger("wandb", self.save_dir)

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size=self.config.opt_step_size)
        opt_state = self.opt_init(self.params)
        wandb.init(
            project="traj_opt", config={"num_steps": self.config.num_steps, "step_size": self.config.opt_step_size}
        )

        best_cost = float("inf")  # Initialize best cost as infinity
        best_params = None  # Initialize best parameters as None

        for i in range(self.config.num_steps):
            current_params = self.get_params(opt_state)
            print("Optimized parameters: ", current_params)
            if self.config.barrier:
                value, opt_state = self.optimization_barrier_step(i, opt_state)
            else:
                value, opt_state = self.optimization_step(i, opt_state)
            val = float(jax.device_get(value))
            wandb.log({"step": i, "value": val})

            if val < best_cost:
                best_cost = val
                best_params = current_params
                print("update best params")

        # Save the best parameters at the end of training
        best_params_file = os.path.join(self.save_dir, "best_" + file_name)
        print(f"Saving best optimized params to: {best_params_file}")
        with open(best_params_file, "wb") as file:
            pickle.dump(best_params, file)

        full_file = os.path.join(self.save_dir, file_name)
        print(f"Saving optimized params to: {full_file}")
        with open(full_file, "wb") as file:
            pickle.dump(self.get_params(opt_state), file)

        wandb.finish()

    def optimization_step(self, step_num, opt_state, clip_value=100.0):
        """Perform one optimization step."""
        grads = jacfwd(self.eval_cost)(self.get_params(opt_state))
        jax.debug.print("grads: {}", grads)
        # Clip gradients
        clipped_grads = jax.tree_util.tree_map(lambda g: jp.clip(g, -clip_value, clip_value), grads)

        # Debugging: print gradients
        jax.debug.print("Clipped grads: {}", clipped_grads)

        opt_state = self.opt_update(step_num, clipped_grads, opt_state)
        return self.eval_cost(self.get_params(opt_state)), opt_state

    def obtain_init(self, optimized_params, env):
        """Obtain the initial configuration for the simulation."""
        if "alpha" in optimized_params and ("q_b_init" in optimized_params and "dq_b_init" in optimized_params):
            q_init, dq_init = env.getBezInitialConfig(
                optimized_params["alpha"], self.env.alpha_base, self.env.step_dur[BehavState.Walking]
            )
            q_init = q_init.at[0:7].set(optimized_params["q_b_init"])
            dq_init = dq_init.at[0:6].set(optimized_params["dq_b_init"])

        if "alpha" in optimized_params and "alpha_base" in optimized_params:
            q_init, dq_init = env.getBezInitialConfig(
                optimized_params["alpha"], optimized_params["alpha_base"], self.env.step_dur[BehavState.Walking]
            )

        elif "alpha" in optimized_params:
            q_init, dq_init = env.getBezInitialConfig(
                optimized_params["alpha"], self.env.alpha_base, self.env.step_dur[BehavState.Walking]
            )

        else:
            q_init = self.env._q_init
            dq_init = self.env._dq_init
        return q_init, dq_init

    def simulate(self, env, optimized_params, num_steps=200, output_video="sim_video.mp4"):
        """Simulate the trajectory using the optimized parameters."""
        q_init, dq_init = self.obtain_init(optimized_params, env)

        jit_env_reset = jit(env.reset)
        jit_env_step = jit(env.step)
        state = jit_env_reset(jax.random.PRNGKey(1), q_init, dq_init, BehavState.Walking)
        state = self.updateParams(state, optimized_params)
        images = []
        rollouts = []

        logged_data_per_step = []  # List to store data for each step
        log_items = ["qpos", "qvel", "qfrc_actuator"]
        reward_items = [
            "ctrl_cost",
            "base_smoothness_reward",
            "jt_smoothness_reward",
            "tracking_joint_reward",
            "grf_penalty",
            "base_smoothness_reward",
            "tracking_lin_vel_reward",
            "tracking_ang_vel_reward",
            "tracking_pos_reward",
            "mechanical_power",
            "cop_reward",
        ]
        env.getRender()

        for _ in range(num_steps):
            state = jit_env_step(state, jp.zeros(self.env.action_size))
            images.append(env.get_image(state.pipeline_state))
            rollouts.append(state)

            # Log the required data
            logged_data = {}
            logged_data = env.log_state_info(
                state.info, ["state", "domain_info", "tracking_err", "joint_desire"], logged_data
            )
            logged_data = env.log_data(state.pipeline_state, log_items, logged_data)
            logged_data = env.log_state_info(state.info["reward_tuple"], reward_items, logged_data)
            logged_data["survival"] = state.done
            logged_data["impact_mismatch"] = state.info["domain_info"]["impact_mismatch"]
            logged_data_per_step.append(logged_data)

        media.write_video(os.path.join(self.save_dir, output_video), images, fps=1.0 / self.env.dt)

        # Plot the logged data
        plot_save_dir = os.path.join(self.save_dir, "plots")
        # env.plot_logged_data(logged_data, plot_save_dir)
        log_file = os.path.join(self.save_dir, "logged_data.json")
        rollout_file = os.path.join(self.save_dir, "rollout_data.pkl")
        with open(log_file, "wb") as f:
            pickle.dump(logged_data_per_step, f)

        with open(rollout_file, "wb") as f:
            pickle.dump(rollouts, f)
        env.plot_logged_data(logged_data_per_step, save_dir=plot_save_dir)
        return

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
        hash_input = str(self.params.keys()) + str(self.cost_terms) + str(self.cost_weights) + str(self.config) + date
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
            "config": self.config.to_dict(),
            "env_config": vars(self.env_config),
        }
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(full_path, exist_ok=True)

        full_config_path = os.path.join(full_path, "config.yaml")
        with open(full_config_path, "w") as yaml_file:
            yaml.dump(config_data, yaml_file, default_flow_style=None)

        return full_path

    def load_config(self, date, base_dir="configurations", folder=None, best_flag=False, export_yaml=True):
        """Load configuration parameters from a file identified by the hash generated with a specific date."""
        hash_path = self.generate_hash(date=date)
        if folder is not None:
            hash_path = folder
        full_path = os.path.join(base_dir, hash_path)

        full_config_path = os.path.join(full_path, "config.yaml")
        with open(full_config_path, "r") as file:
            config = yaml.safe_load(file)
        self.save_dir = full_path

        if best_flag:
            file_name = "best_optimized_params.pkl"
        else:
            file_name = "optimized_params.pkl"
        file = os.path.join(self.save_dir, file_name)
        print(f"Loading config from: {file}")

        with open(file, "rb") as optimized_File:
            config["optimized_params"] = pickle.load(optimized_File)
        print(f"optimized_params: {config['optimized_params']}")

        if os.path.exists(file_name):
            print(f"File:{file_name} exists")

        if "alpha" in config["optimized_params"] and export_yaml:
            gait_params_file = os.path.join(ROOT, "..", "models", "exo", self.env_config.jt_traj_file)
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
    config.rand_terrain = False
    config.traj_opt = True
    config.impact_based_switching = True
    config.impact_threshold = 400.0
    config.jt_traj_file = "jt_bez_2023-09-10.yaml"
    config.physics_steps_per_control_step = 5
    config.reset_noise_scale = 1e-2
    config.no_noise = True
    config.controller.hip_regulation = False
    config.controller.cop_regulation = False
    env = Exo_TSC(config)
    param_keys = ["alpha"]
    # old_config = "configurations/ddf5bdf083e2f5daf2b7c75b01626c7591598440febd83cc7b9026d4ce99dff9/optimized_params.pkl"
    # with open(old_config, "rb") as file:
    #     init_guess = pickle.load(file)
    # param_values = [
    #     init_guess["alpha"],
    #     init_guess["hip_regulator_gain"],
    #     env.config.impact_threshold,
    # ]  # Example initial conditions
    param_values = [env.alpha]  # Example initial conditions

    # to do add grf penalty, and check the impact_mismatch

    cost_terms = [
        "tracking_err",
        "output_err",
        "survival",
    ]

    cost_weights = {
        "tracking_err": 1.0,
        "output_err": 1.0,
        "survival": 100.0,
    }

    traj_opt_config = TrajOptConfig()
    traj_opt_config.barrier = False
    optimizer = TrajectoryOptimizer(config, traj_opt_config, param_keys, param_values, cost_terms, cost_weights)
    # filename = "multi_no_pos_optimized_params_update_v2.pkl" #either this or without v2 is the old result

    # retrain 279b399b0aa77c3fd4a87f616c7dbc15bb8afc6d356975679f426d1b8c672d94
    # --> saved at ddf5bdf083e2f5daf2b7c75b01626c7591598440febd83cc7b9026d4ce99dff9

    # a1ff6cdd
    # 9666, work multiple steps in c++

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
        optimizer.train()
    else:
        # config = ExoConfig()
        # config.rand_terrain = True
        # config.impact_based_switching = True
        # config.impact_threshold = 200.0
        # config.jt_traj_file = "default_bez.yaml"
        # config.physics_steps_per_control_step = 5
        # config.reset_noise_scale = 1e-2
        config.traj_opt = False
        config.no_noise = True
        config.controller.hip_regulation = False
        config.controller.cop_regulation = False
        env = Exo(config)

        param_date = "20240118"
        # folder = "04cb3d3b2992140e9c9dcbaa9422e7b5dd5b1bde392f1f45cbfdf40a1ca272e9"
        opt_config = optimizer.load_config(date=param_date, best_flag=True)
        # nominal = {"cop_regulator_gain": env.config.controller.cop_regulator_gain}
        nominal = dict(zip(param_keys, param_values))

        # ,"hip_regulator_gain":env.config.controller.hip_regulator_gain,"impact_threshold":env.config.impact_threshold
        # optimizer.simulate(
        #     env,
        #     opt_config["optimized_params"],
        #     num_steps=1000,
        #     output_video="mjx_traj_opt_plane_" + param_date + ".mp4",
        # )

        optimizer.simulate(env, nominal, num_steps=2400, output_video="nominal_box_" + param_date + ".mp4")
