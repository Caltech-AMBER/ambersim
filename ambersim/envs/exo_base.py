import enum
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import mediapy as media
import mujoco as mj
import numpy as np
import yaml
from brax.base import Base, Motion, Transform
from brax.envs.base import PipelineEnv, State
from jax import lax
from jax.scipy.integrate import trapezoid
from mujoco import _enums, _structs, mjx
from mujoco.mjx._src import scan
from mujoco.mjx._src.types import DisableBit

from ambersim import ROOT
from ambersim.base import MjxEnv
from ambersim.utils.io_utils import set_actuators_type


class BehavState(enum.IntEnum):
    """Behavioral states of the exoskeleton."""

    ToLoading = 0
    Loading = 1
    ToStopped = 2
    Stopped = 3
    WantToStart = 4
    StartingStep = 5
    Walking = 6
    WantToStop = 7
    StoppingStep = 8


class StanceState(enum.IntEnum):
    """Stance states of the exoskeleton."""

    InTheAir = -1
    Left = 0
    Right = 1
    DS = 2
    LeftDS = 3
    RightDS = 4


@dataclass
class ExoRewardConfig:
    """Weightings for the reward function."""

    # Tracking rewards are computed using exp(-delta^2/sigma)
    # sigma can be a hyperparameters to tune.
    # Track the base x-y velocity (no z-velocity tracking.)
    tracking_lin_vel: float = 0.5

    # Track the angular velocity along z-axis, i.e. yaw rate.
    tracking_ang_vel: float = 0.5

    # Below are regularization terms, we roughly divide the
    # terms to base state regularizations, joint
    # regularizations, and other behavior regularizations.
    # Penalize the base velocity in z direction, L2 penalty.
    tracking_base_ori: float = 1.0

    # Penalize the base roll and pitch rate. L2 penalty.
    tracking_base_pos: float = 1.0

    tracking_joint: float = 0.3
    # Penalize non-zero roll and pitch angles. L2 penalty.
    # orientation: float = -5.0
    tracking_sigma_vel: float = 0.5
    tracking_sigma_pos: float = 0.5
    tracking_sigma_joint_pos: float = 0.2

    # grf penalty
    grf_cost_weight: float = -0.01
    # L2 regularization of joint torques, |tau|^2.
    ctrl_cost_weight: float = -1e-10

    unhealthy_penalty: float = -100.0

    healthy_reward: float = 10.0


class ExoConfig:
    """config dataclass that specified the reward coefficient and other custom setting for exo env."""

    reward: ExoRewardConfig = ExoRewardConfig()
    terminate_when_unhealthy: bool = True
    reset_2_stand: bool = False
    healthy_z_range: tuple = (0.85, 1)
    desired_cop_range: tuple = (-0.1, 0.1)
    reset_noise_scale: float = 1e-5
    history_size: float = 5
    xml_file: str = "loadedExo.xml"
    jt_traj_file: str = "jt_bez_2023-09-10.yaml"
    loading_pos_file: str = "sim_config_loadingPos.yaml"
    ctrl_limit_file: str = "limits.yaml"
    rand_terrain: bool = False
    position_ctrl: bool = True
    residual_action_space: bool = False
    physics_steps_per_control_step: int = 10
    action_scale: float = 0.01
    custom_action_space: jp.ndarray = jp.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1])
    custom_act_idx: jp.ndarray = jp.array([4, 5, 10, 11])


class Exo(MjxEnv):
    """custom environment for the exoskeleton."""

    def __init__(self, config: ExoConfig = ExoConfig, **kwargs):
        """Initialize the environment."""
        self.config = config
        if config.rand_terrain:
            xml_file = "loadedExo_no_terrain.xml"
            path = os.path.join(ROOT, "models", "exo", xml_file)
            # path = self.genRandBoxes(path,grid_size = (2,5))
        else:
            path = os.path.join(ROOT, "..", "models", "exo", config.xml_file)

        self.model = mj.MjModel.from_xml_path(path)
        self.load_traj()
        self.load_ctrl_params()

        self.loadRelevantParams()

        if self.config.residual_action_space:
            self.custom_action_space = config.custom_action_space
            self.custom_act_space_size = int(jp.sum(self.custom_action_space))
        else:
            self.custom_action_space = jp.ones(self.model.nu)
            self.custom_act_space_size = self.model.nu
        # calculate observation size
        # TODO: fix hard code here
        self.observation_size_single_step = self.model.nq + self.model.nv + 8 + self.model.nu + 3

        super().__init__(mj_model=self.model, physics_steps_per_control_step=self.config.physics_steps_per_control_step)

    @property
    def action_size(self) -> int:
        """Override the super class action size function."""
        return self.custom_act_space_size

    def loadRelevantParams(self):
        """Load the relevant parameters for the environment."""
        self.base_frame_idx = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "torso")
        foot_geom_idx = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "right_sole"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "left_sole"),
        ]
        self.foot_geom_idx = jp.array(foot_geom_idx)

    def load_traj(self) -> None:
        """Load default trajectory from yaml file specfied in the config."""
        gait_params_file = os.path.join(ROOT, "..", "models", "exo", self.config.jt_traj_file)
        with open(gait_params_file, "r") as file:
            gait_params = yaml.safe_load(file)

        self.step_dur = 0.5 * jp.ones(9)
        self.step_dur = self.step_dur.at[BehavState.Walking].set(gait_params["step_dur"])
        self.step_dur = self.step_dur.at[BehavState.StoppingStep].set(gait_params["step_dur"])
        self.step_dur = self.step_dur.at[BehavState.StartingStep].set(gait_params["step_dur"])

        self.step_start = 0.0
        coeff_jt = np.reshape(np.array(gait_params["coeff_jt"]), (12, 8), order="F")
        coeff_b = np.reshape(np.array(gait_params["coeff_b"]), (6, 8), order="F")
        self.alpha = jp.array(coeff_jt)
        self.alpha_base = jp.array(coeff_b)
        self.bez_deg = 7
        self._q_init = jp.concatenate([jp.array(gait_params["ffPos"]), jp.array(gait_params["startingPos"])], axis=0)
        self._dq_init = jp.concatenate([jp.array(gait_params["ffVel"]), jp.array(gait_params["startingVel"])], axis=0)

        R = self._remap_coeff()
        self.R = R[-self.model.nu :, -self.model.nu :]
        self.R_base = R[:6, :6]
        self.q_desire = self._q_init[-self.model.nu :]
        loading_pos_file = os.path.join(ROOT, "..", "models", "exo", self.config.loading_pos_file)
        with open(loading_pos_file, "r") as file:
            load_params = yaml.safe_load(file)

        self._q_load = jp.concatenate([jp.array(load_params["ffPos"]), jp.array(load_params["startingPos"])], axis=0)
        q_start = jp.concatenate(
            [jp.array(load_params["ffPosStartingStep"]), jp.array(load_params["startingStepPos"])], axis=0
        )
        q_stop = jp.concatenate(
            [jp.array(load_params["ffPosStartingStep"]), jp.array(load_params["stoppingPos"])], axis=0
        )

        max_state = max(BehavState, key=lambda member: member.value)
        self._q_default = jp.zeros((max_state.value + 1, self.model.nq))
        self._dq_default = jp.zeros((max_state.value + 1, self.model.nv))

        self._q_default = self._q_default.at[BehavState.ToLoading.value, :].set(self._q_load)
        self._q_default = self._q_default.at[BehavState.Loading.value, :].set(self._q_load)
        self._q_default = self._q_default.at[BehavState.WantToStart.value, :].set(q_start)

        self._q_default = self._q_default.at[BehavState.WantToStop.value, :].set(q_stop)
        self._q_default = self._q_default.at[BehavState.ToStopped.value, :].set(q_stop)
        self._q_default = self._q_default.at[BehavState.Stopped.value, :].set(q_stop)
        self._q_default = self._q_default.at[BehavState.StoppingStep.value, :].set(q_stop)

        self._q_default = self._q_default.at[BehavState.Walking.value, :].set(self._q_init)
        self._dq_default = self._dq_default.at[BehavState.Walking.value, :].set(self._dq_init)

    def update_default_traj(self, new_traj: jp.ndarray, step_dur: float) -> None:
        """Update the default trajectory with the new trajectory."""
        self.alpha = new_traj
        self.step_dur = step_dur
        self._q_init = self._q_init.at[-self.model.nu :].set(
            self._forward(self.step_start, self.step_start, self.step_dur, self.alpha)
        )
        self._dq_init = self._dq_init.at[-self.model.nu :].set(
            self._forward_vel(self.step_start, self.step_start, self.step_dur, self.alpha)
        )

    def load_ctrl_params(self) -> None:
        """Load joint limit and config limit from yaml file specfied in the config."""
        limit_file = os.path.join(ROOT, "..", "models", "exo", self.config.ctrl_limit_file)
        with open(limit_file, "r") as file:
            ctrl_limit = yaml.safe_load(file)
        self._p_gains = jp.array(ctrl_limit["joint"]["kp"])
        self._d_gains = jp.array(ctrl_limit["joint"]["kd"])

        self._jt_lb = jp.array(ctrl_limit["joint"]["min"])
        self._jt_ub = jp.array(ctrl_limit["joint"]["max"])

        torque_limits = jp.array(ctrl_limit["torque"])

        self._torque_lb = -torque_limits
        self._torque_ub = torque_limits

        max_state = max(BehavState, key=lambda member: member.value)
        self._blend_dur = 0.2 * jp.ones((max_state.value + 1, 1))
        self._blend_dur = self._blend_dur.at[BehavState.StartingStep.value].set(0.8)
        self._blend_dur = self._blend_dur.at[BehavState.ToLoading.value].set(1)
        self._blend_dur = self._blend_dur.at[BehavState.WantToStart.value].set(0.5)

        if self.config.position_ctrl:
            for jt_idx in range(self.model.nu):
                self.model = set_actuators_type(self.model, "position", jt_idx, kp=self._p_gains[jt_idx])

    def update_pd_gains(self) -> None:
        """Update the pd gains for the joints."""
        for jt_idx in range(self.model.nu):
            self.model = set_actuators_type(
                self.model, "position", jt_idx, kp=self._p_gains[jt_idx], kd=self._d_gains[jt_idx]
            )

    def reset_bez(self, rng: jp.ndarray, alpha: jp.ndarray, step_dur: float, state: BehavState) -> State:
        """Reset the environment with the bez for a given trajectory."""
        self._q_init = self._q_init.at[-self.model.nu :].set(
            self._forward(self.step_start, self.step_start, step_dur, alpha)
        )
        self._dq_init = self._dq_init.at[-self.model.nu :].set(
            self._forward_vel(self.step_start, self.step_start, step_dur, alpha)
        )
        return self.reset(rng, state)

    def reset(self, rng: jp.ndarray, behavstate: BehavState) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self.config.reset_noise_scale, self.config.reset_noise_scale

        qpos = self._q_default[behavstate, :] + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = self._dq_default[behavstate, :] + jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        reward, done, zero = jp.zeros(3)

        state_info = {
            "state": behavstate,
            "offset": jp.zeros(12),
            "domain_info": {
                "step_start": 0.0,
                "domain_idx": 0.0,
            },
            "mechanical_power": zero,
            "obs_history": jp.zeros(self.config.history_size * self.observation_size_single_step),
            "nominal_action": jp.zeros(12),
            "blended_action": jp.zeros(12),
            "base_pos_desire": jp.zeros(6),
            "base_vel_desire": jp.zeros(6),
            "joint_desire": jp.zeros(12),
            "last_action": jp.zeros(12),
            "tracking_err": jp.zeros(12),
            "reward_tuple": {
                "ctrl_cost": zero,
                "tracking_lin_vel_reward": zero,
                "tracking_ang_vel_reward": zero,
                "tracking_pos_reward": zero,
                "tracking_orientation_reward": zero,
                "trtacking_joint_reward": zero,
                "grf_penalty": zero,
            },
        }

        obs = self._get_obs(data, jp.zeros(self.action_size), state_info)

        metrics = {"total_dist": 0.0}
        for k in state_info["reward_tuple"]:
            metrics[k] = state_info["reward_tuple"][k]

        return State(data, obs, reward, done, metrics, state_info)

    def getFootPos(self, state: State) -> jp.ndarray:
        """Get the foot position in world frame."""
        data = state.pipeline_state
        foot_pos = data.geom_xpos[self.foot_geom_idx, 0:3]
        return foot_pos

    def bez_step(self, state: State, alpha: jp.ndarray, step_dur: float) -> State:
        """Run one step of the environment with the bez trajectory."""
        data0 = state.pipeline_state
        step_start = state.info["domain_info"]["step_start"]
        # action = self._forward(data0.time, step_start, step_dur, alpha)

        behavstate = state.info["state"]

        def true_fun(_):
            return self._forward(data0.time, step_start, step_dur, alpha)

        def false_fun(_):
            return self._q_default[behavstate, -self.model.nu :]

        # behavstate = BehavState.Walking
        action = lax.cond(behavstate == BehavState.Walking, true_fun, false_fun, None)
        state.info["joint_desire"] = action

        if self.config.position_ctrl:
            blended_action = self.jt_blending(data0.time, self.step_dur[behavstate], action, state.info)
            motor_targets = jp.clip(blended_action, self._jt_lb, self._jt_ub)
        else:
            Warning("torque control is not implemented yet")
        data = self.pipeline_step(data0, motor_targets)

        # observation data
        obs = self._get_obs(data, action, state.info)
        reward_tuple = self.evaluate_reward(data0, data, jp.zeros(12), state.info)

        # state management
        state.info["reward_tuple"] = reward_tuple
        state.info["last_action"] = jp.zeros(12)
        state.info["blended_action"] = blended_action
        state.info["mechanical_power"] = self.mechanical_power(data)
        state.info["tracking_err"] = (
            data.qpos[-self.model.nu :] - motor_targets
        )  # currently assuming motor_targets is the desired joint angles; TODO handle torque case

        # resetting logic if joint limits are reached or robot is falling
        # violate joint limit
        done = self.checkDone(data)

        reward = jp.sum(jp.array(list(reward_tuple.values())))
        reward = reward + (1 - done) * self.config.reward.healthy_reward

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def checkDone(self, data: mjx.Data) -> float:
        """Check if the robot is falling or joint limits are reached."""
        # resetting logic if joint limits are reached or robot is falling
        # violate joint limit
        joint_angles = data.qpos[-self.model.nu :]
        done = 0.0
        done = jp.where(
            jp.logical_or(
                jp.any(joint_angles < 0.97 * self._jt_lb),
                jp.any(joint_angles > 0.97 * self._jt_ub),
            ),
            1.0,
            done,
        )
        # no ground reaction force
        done = jp.where(jp.sum(self._get_contact_force(data)) < 10.0, 1.0, done)

        # falling
        done = jp.where(data.qpos[2] < 0.7, 1.0, done)
        return done

    def getNominalDesire(self, state: State) -> State:
        """Get the nominal desire for the current state."""
        q_desire, state = self.getWalkingNomDes(state)
        behavstate = state.info["state"]

        def true_fun(_):
            return q_desire

        def false_fun(_):
            return self._q_default[behavstate, -self.model.nu :]

        # behavstate = BehavState.Walking
        q_desire = lax.cond(behavstate == BehavState.Walking, true_fun, false_fun, None)
        return q_desire, state

    def getWalkingNomDes(self, state: State) -> jp.ndarray:
        """Get the nominal desire for walking."""
        data0 = state.pipeline_state
        domain_idx = state.info["domain_info"]["domain_idx"]
        step_start = state.info["domain_info"]["step_start"]

        # action = self.conv_action_based_on_idx(cur_action, jp.zeros(12))
        def update_step(step_start, domain_idx):
            new_step_start = data0.time
            new_domain_idx = 1 - domain_idx  # Switch domain_idx between 0 and 1

            return new_step_start, new_domain_idx

        def no_update(step_start, domain_idx):
            return step_start, domain_idx

        condition = (data0.time - step_start) / self.step_dur[BehavState.Walking] >= 1
        new_step_start, domain_idx = lax.cond(
            condition, lambda args: update_step(*args), lambda args: no_update(*args), (step_start, domain_idx)
        )

        step_start = new_step_start
        alpha = lax.cond(domain_idx == 0, lambda _: self.alpha, lambda _: jp.dot(self.R, self.alpha), None)
        alpha_base = lax.cond(
            domain_idx == 0, lambda _: self.alpha_base, lambda _: jp.dot(self.R_base, self.alpha_base), None
        )
        q_desire = self._forward(data0.time, step_start, self.step_dur[BehavState.Walking], alpha)

        state.info["nominal_action"] = q_desire
        state.info["base_pos_desire"] = self._forward(
            data0.time, step_start, self.step_dur[BehavState.Walking], alpha_base
        )
        state.info["base_vel_desire"] = self._forward_vel(
            data0.time, step_start, self.step_dur[BehavState.Walking], alpha_base
        )
        state.info["domain_info"]["domain_idx"] = domain_idx
        state.info["domain_info"]["step_start"] = step_start

        return q_desire, state

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        cur_action = action

        data0 = state.pipeline_state

        q_desire, state = self.getNominalDesire(state)

        action = self.config.action_scale * action + q_desire
        state.info["joint_desire"] = action

        if self.config.position_ctrl:
            # jax.debug.print("nominal action: {}", action)
            blended_action = self.jt_blending(data0.time, self.step_dur[state.info["state"]], action, state.info)
            # jax.debug.print("blended action: {}", blended_action)
            motor_targets = jp.clip(blended_action, self._jt_lb, self._jt_ub)
        else:
            motor_targets = jp.clip(action, self._torque_lb, self._torque_ub)

        data = self.pipeline_step(data0, motor_targets)

        # observation data
        obs = self._get_obs(data, action, state.info)
        reward_tuple = self.evaluate_reward(data0, data, cur_action, state.info)

        # state management
        state.info["reward_tuple"] = reward_tuple
        state.info["last_action"] = cur_action
        state.info["blended_action"] = blended_action
        state.info["mechanical_power"] = self.mechanical_power(data)
        state.info["tracking_err"] = (
            data.qpos[-self.model.nu :] - action
        )  # currently assuming motor_targets is the desired joint angles; TODO handle torque case

        # resetting logic if joint limits are reached or robot is falling
        # violate joint limit
        done = self.checkDone(data)

        reward = jp.sum(jp.array(list(reward_tuple.values())))
        reward = reward + (1 - done) * self.config.reward.healthy_reward

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def state_condition_met(self, maxErrorTrigger, minTransitionTime, data: mjx.Data, state_info: Dict[str, Any]):
        """Check if the error is within the threshold and minimum time has passed since the last transition."""
        # Check if the error is within the threshold and minimum time has passed since the last transition
        step_start = state_info["domain_info"]["step_start"]
        err = state_info["tracking_err"]
        current_time = data.time
        return jp.max(jp.abs(err)) <= maxErrorTrigger and (current_time - step_start) >= minTransitionTime

    def log_data(self, state, log_items, logged_data):
        """Logs specified items from the state."""
        # logged_data = {}
        for item in log_items:
            if hasattr(state, item):
                logged_data[item] = getattr(state, item).tolist()  # Convert to list if it's a NumPy array
            else:
                print(f"Warning: '{item}' not found in state")
        return logged_data

    def log_state_info(self, state_info, log_items, logged_data):
        """Logs specified items from the state_info."""
        for item in log_items:
            if item in state_info:
                logged_data[item] = state_info[item]
            else:
                print(f"Warning: '{item}' not found in state_info")
        return logged_data

    def plot_attribute(self, attribute, values, save_dir):
        """Creates and saves subplots for each dimension of an attribute.

        Param attribute: The name of the attribute.
        Param values: The values of the attribute.
        Param save_dir: Directory to save the plot.
        """
        num_dims = len(values[0]) if isinstance(values[0], list) else 1
        plt.figure(figsize=(10, 4 * num_dims))

        for dim in range(num_dims):
            ax = plt.subplot(num_dims, 1, dim + 1)
            if num_dims > 1:
                dim_values = [v[dim] for v in values]
                ax.plot(dim_values)
                ax.set_title(f"{attribute} - Dimension {dim}")
            else:
                ax.plot(values)
                ax.set_title(attribute)

            ax.set_xlabel("Time step")
            ax.set_ylabel(attribute)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{attribute}.png"))
        plt.close()

    def plot_logged_data(self, logged_data, save_dir="plots"):
        """Plotting function for the logged data."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Determine the number of unique attributes to plot
        # attributes = set(key for data_point in logged_data for key in data_point.keys())
        attributes = {key for data_point in logged_data for key in data_point.keys()}

        # Create and save a plot for each attribute
        for attribute in attributes:
            values = [
                data_point[attribute]
                for data_point in logged_data
                if attribute in data_point and not isinstance(data_point[attribute], dict)
            ]

            if values:
                self.plot_attribute(attribute, values, save_dir)

    def process_rollouts(self, rollouts_dir, plot_save_dir="plots"):
        """Processes saved rollouts, extracting and plotting logged data.

        :param rollouts_dir: Directory containing the rollout files.
        :param plot_save_dir: Directory to save the plots.
        """
        if not os.path.exists(rollouts_dir):
            print(f"Rollouts directory {rollouts_dir} not found.")
            return

        for filename in os.listdir(rollouts_dir):
            if filename.endswith(".pkl"):  # Assuming the files are pickle files
                log_file_path = os.path.join(rollouts_dir, filename)
                print(f"Processing {filename}...")
                self.load_and_plot_data(log_file_path, plot_save_dir)

    def run_sim_from_standing(
        self, rng, num_steps=400, log_file="simulation_times.log", output_video="nominal_policy_from_standing_video.mp4"
    ):
        """Run the simulation from standing position."""
        # start from standing, wait for 0.5 second, go to startingpos, wait for 0.5 second, start walking, walk 2 steps, and then go to stopping pos
        self.getRender()

        jit_reset = jax.jit(self.reset)
        jit_step = jax.jit(self.step)

        # currentState = BehavState.ToLoading
        state = jit_reset(rng, BehavState.WantToStart)
        images = []

        maxErrorTrigger = 0.01
        minTransitionTime = 0.05

        log_items = ["qpos", "qvel", "qfrc_actuator"]
        logged_data_per_step = []

        prev_domain = state.info["domain_info"]["domain_idx"]

        rollouts = []

        with open(log_file, "w") as file:
            for _ in range(num_steps):
                start_time = time.time()
                state = jit_step(state, jp.zeros(12))  # Replace with your control strategy
                end_time = time.time()

                step_duration = end_time - start_time

                # Log the time taken for each step
                logged_data = {}
                logged_data = self.log_data(state.pipeline_state, log_items, logged_data)
                logged_data = self.log_state_info(
                    state.info, ["state", "domain_info", "tracking_err", "joint_desire"], logged_data
                )
                logged_data_per_step.append(logged_data)
                # Log the time taken for each step
                file.write(f"Step duration: {step_duration}\n")

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
                        jax.debug.print(
                            "starting pos: {}", self._q_default[BehavState.WantToStart.value, -self.model.nu :]
                        )
                        state_change = True

                elif state.info["state"] == BehavState.Walking:
                    jax.debug.print("state: Walking")
                    jax.debug.print("blended_action: {}", state.info["blended_action"])
                    jax.debug.print("joint_desire: {}", state.info["joint_desire"])
                    if state.info["domain_info"]["domain_idx"] != prev_domain:
                        state.info["offset"] = (
                            state.pipeline_state.qpos[-self.model.nu :] - self.getNominalDesire(state)[0]
                        )
                        jax.debug.print("offset: {}", state.info["offset"])
                        prev_domain = state.info["domain_info"]["domain_idx"]
                        jax.debug.print("domain changed to {}", state.info["domain_info"]["domain_idx"])
                        # state_change = True

                if state_change:
                    state.info["domain_info"]["step_start"] = state.pipeline_state.time
                    state.info["offset"] = state.pipeline_state.qpos[-self.model.nu :] - self.getNominalDesire(state)[0]

                    jax.debug.print("offset: {}", state.info["offset"])

                rollouts.append(state)
                images.append(self.get_image(state.pipeline_state))

        media.write_video(output_video, images, fps=1.0 / self.dt)

        log_file = "logged_data.json"
        rollout_file = "rollout_data.pkl"
        with open(log_file, "wb") as f:
            pickle.dump(logged_data_per_step, f)

        with open(rollout_file, "wb") as f:
            pickle.dump(rollouts, f)
        self.plot_logged_data(logged_data_per_step)
        # check if

        return

    def run_base_sim(
        self, rng, num_steps=400, log_file="simulation_times.log", output_video="nominal_policy_video.mp4"
    ):
        """Run the simulation basic version."""
        self.getRender()
        jit_reset = jax.jit(self.reset)
        jit_step = jax.jit(self.step)

        state = jit_reset(rng, BehavState.Walking)
        prev_state = state.info["state"]
        prev_domain = state.info["domain_info"]["domain_idx"]
        images = []

        with open(log_file, "w") as file:
            for _ in range(num_steps):
                start_time = time.time()
                state = jit_step(state, jp.zeros(12))  # Replace with your control strategy
                state.info["state"] = BehavState.Walking

                # check previous state
                if state.info["state"] != prev_state or state.info["domain_info"]["domain_idx"] != prev_domain:
                    state.info["offset"] = state.info["tracking_err"]
                    # jax.debug.print("offset: {}", state.info['offset'])
                    prev_state = state.info["state"]
                    prev_domain = state.info["domain_info"]["domain_idx"]
                    # jax.debug.print("state changed to {}", state.info["state"])
                    # jax.debug.print("domain changed to {}", state.info["domain_info"]["domain_idx"])
                    # domain update should be handled inside

                end_time = time.time()

                step_duration = end_time - start_time

                # Log the time taken for each step
                file.write(f"Step duration: {step_duration}\n")

                images.append(self.get_image(state.pipeline_state))

        media.write_video(output_video, images, fps=1.0 / self.dt)
        return

    def run_bez_sim_from_standing(
        self,
        rng,
        alpha,
        step_dur,
        num_steps=400,
        log_file="simulation_times.log",
        output_video="bez_policy_from_standing_video.mp4",
    ):
        """Run the simulation from standing position with given bezier trajectory."""
        # start from standing, wait for 0.5 second, go to startingpos, wait for 0.5 second, start walking, walk 2 steps, and then go to stopping pos
        self.getRender()

        jit_reset = jax.jit(self.reset)
        jit_step = jax.jit(self.bez_step)

        # currentState = BehavState.ToLoading
        state = jit_reset(rng, BehavState.WantToStart)
        images = []

        maxErrorTrigger = 0.01
        minTransitionTime = 0.05

        log_items = ["qpos", "qvel", "qfrc_actuator"]
        logged_data_per_step = []

        # prev_domain = state.info["domain_info"]["domain_idx"]

        rollouts = []

        with open(log_file, "w") as file:
            for _ in range(num_steps):
                start_time = time.time()
                state = jit_step(state, alpha, step_dur)
                end_time = time.time()

                step_duration = end_time - start_time

                # Log the time taken for each step
                logged_data = {}
                logged_data = self.log_data(state.pipeline_state, log_items, logged_data)
                logged_data = self.log_state_info(
                    state.info, ["state", "domain_info", "tracking_err", "joint_desire"], logged_data
                )
                logged_data_per_step.append(logged_data)
                # Log the time taken for each step
                file.write(f"Step duration: {step_duration}\n")

                state_change = False

                # #check state transition

                if state.info["state"] == BehavState.WantToStart:
                    minTransitionTime = 2 * self.step_dur[BehavState.WantToStart]
                    if self.state_condition_met(maxErrorTrigger, minTransitionTime, state.pipeline_state, state.info):
                        state.info["state"] = BehavState.Walking
                        state_change = True

                elif state.info["state"] == BehavState.Walking:
                    jax.debug.print("state: Walking")
                    jax.debug.print("blended_action: {}", state.info["blended_action"])
                    jax.debug.print("joint_desire: {}", state.info["joint_desire"])
                    jax.debug.print("current joint config: {}", state.pipeline_state.qpos[-self.model.nu :])
                    jax.debug.print("time: {}", state.pipeline_state.time)
                    jax.debug.print(
                        "time - step_start: {}", state.pipeline_state.time - state.info["domain_info"]["step_start"]
                    )
                    if (state.pipeline_state.time - state.info["domain_info"]["step_start"]) / step_dur > 1:
                        state.info["domain_info"]["step_start"] = state.pipeline_state.time

                        alpha = jp.dot(self.R, alpha)
                        print("update step_start")

                        state.info["offset"] = state.pipeline_state.qpos[-self.model.nu :] - self._forward(
                            state.pipeline_state.time, state.info["domain_info"]["step_start"], step_dur, alpha
                        )
                        jax.debug.print("offset: {}", state.info["offset"])
                        # prev_domain = state.info["domain_info"]["domain_idx"]
                        # jax.debug.print("domain changed to {}", state.info["domain_info"]["domain_idx"])
                        # state_change = True

                if state_change:
                    state.info["domain_info"]["step_start"] = state.pipeline_state.time
                    state.info["offset"] = state.pipeline_state.qpos[-self.model.nu :] - self.getNominalDesire(state)[0]

                    jax.debug.print("offset: {}", state.info["offset"])

                rollouts.append(state)
                images.append(self.get_image(state.pipeline_state))

        media.write_video(output_video, images, fps=1.0 / self.dt)

        log_file = "logged_data.json"
        rollout_file = "rollout_data.pkl"
        with open(log_file, "wb") as f:
            pickle.dump(logged_data_per_step, f)

        with open(rollout_file, "wb") as f:
            pickle.dump(rollouts, f)
        self.plot_logged_data(logged_data_per_step)
        # check if
        return

    def run_base_bez_sim(
        self, rng, alpha, step_dur, num_steps=400, log_file="bez_sim.log", output_video="bez_video.mp4"
    ):
        """Run the simulation basic version with given bezier trajectory."""
        self.getRender()
        jit_reset = jax.jit(self.reset_bez)
        jit_step = jax.jit(self.bez_step)

        state = jit_reset(rng, alpha, step_dur, BehavState.Walking)
        images = []

        with open(log_file, "w") as file:
            for _ in range(num_steps):
                start_time = time.time()
                state = jit_step(state, alpha, step_dur)

                end_time = time.time()

                step_duration = end_time - start_time

                # Log the time taken for each step
                file.write(f"Step duration: {step_duration}\n")

                if (state.pipeline_state.time - state.info["domain_info"]["step_start"]) / step_dur > 1:
                    state.info["domain_info"]["step_start"] = state.pipeline_state.time

                    alpha = jp.dot(self.R, alpha)
                    print("update step_start")

                images.append(self.get_image(state.pipeline_state))

        media.write_video(output_video, images, fps=1.0 / self.dt)
        return

    def evaluate_reward(self, data0: mjx.Data, data: mjx.Data, action: jp.ndarray, state_info: Dict[str, Any]) -> dict:
        """Evaluate the reward function."""

        def eulRates2omega(eulRates, orientation):
            # Input processing
            assert len(eulRates) == 3, "The omega must be a numerical vector of length 3."

            x, y, z = orientation

            M = jp.array(
                [[jp.cos(y) * jp.cos(z), jp.sin(z), 0], [-jp.cos(y) * jp.sin(z), jp.cos(z), 0], [jp.sin(y), 0, 1]]
            )

            omega = jp.dot(M, eulRates)

            return omega

        def quat2eulXYZ(quat):
            # Should assume wxyz convention
            # z1_2 should be qy, 1 should be x, w should be 0

            eul = jp.zeros(3, dtype=jp.float32)
            b = 1.0 / jp.sqrt(((quat[0] * quat[0] + quat[1] * quat[1]) + quat[2] * quat[2]) + quat[3] * quat[3])
            z1_idx_0 = quat[0] * b
            z1_idx_1 = quat[1] * b
            z1_idx_2 = quat[2] * b
            b *= quat[3]
            aSinInput = 2.0 * (z1_idx_1 * b + z1_idx_2 * z1_idx_0)

            # Ensure aSinInput is within [-1.0, 1.0]
            aSinInput = jp.clip(aSinInput, -1.0, 1.0)

            eul_tmp = z1_idx_0 * z1_idx_0
            b_eul_tmp = z1_idx_1 * z1_idx_1
            c_eul_tmp = z1_idx_2 * z1_idx_2
            d_eul_tmp = b * b

            eul = jp.array(
                [
                    jp.arctan2(
                        -2.0 * (z1_idx_2 * b - z1_idx_1 * z1_idx_0), ((eul_tmp - b_eul_tmp) - c_eul_tmp) + d_eul_tmp
                    ),
                    jp.arcsin(aSinInput),
                    jp.arctan2(
                        -2.0 * (z1_idx_1 * z1_idx_2 - b * z1_idx_0), ((eul_tmp + b_eul_tmp) - c_eul_tmp) - d_eul_tmp
                    ),
                ]
            )

            return eul

        # convert relevant fields
        eul = quat2eulXYZ(data.qpos[3:7])
        eul_rate = eulRates2omega(data.qvel[3:6], eul)

        # base coordinate tracking
        domain_idx = state_info["domain_info"]["domain_idx"]
        # base_pos = data.qpos[0:3] - data.geom_xpos[self.foot_geom_idx[domain_idx],0:3]

        def get_base_pos(domain_idx, data, foot_geom_idx):
            base_pos = data.qpos[0:3] - data.geom_xpos[foot_geom_idx[domain_idx], 0:3]
            return base_pos

        base_pos = jax.lax.cond(
            domain_idx == 0,
            lambda _: get_base_pos(0, data, self.foot_geom_idx),
            lambda _: get_base_pos(1, data, self.foot_geom_idx),
            operand=None,
        )

        def get_stance_contact(idx, data):
            return data.efc_force[data.contact.efc_address[idx]]

        stance_grf = jax.lax.cond(
            domain_idx == 0,
            lambda _: get_stance_contact([4, 5, 6, 7], data),
            lambda _: get_stance_contact([0, 1, 2, 3], data),
            operand=None,
        )

        grf_penalty = self.config.reward.grf_cost_weight * (1.0 - jp.sum(stance_grf) / (self.mass * 9.81))

        tracking_pos_reward = self.config.reward.tracking_base_pos * jp.exp(
            -jp.sum(jp.square(base_pos - state_info["base_pos_desire"][0:3])) / self.config.reward.tracking_sigma_pos
        )

        tracking_orientation_reward = self.config.reward.tracking_base_ori * jp.exp(
            -jp.sum(jp.square(eul - state_info["base_pos_desire"][3:6])) / self.config.reward.tracking_sigma_pos
        )

        tracking_joint_reward = self.config.reward.tracking_joint * jp.exp(
            -jp.sum(jp.square(data.qpos[-self.model.nu :] - state_info["nominal_action"]))
            / self.config.reward.tracking_sigma_joint_pos
        )

        tracking_ang_vel_reward = self.config.reward.tracking_ang_vel * jp.exp(
            -jp.sum(jp.square(eul_rate - state_info["base_vel_desire"][3:6])) / self.config.reward.tracking_sigma_vel
        )

        tracking_lin_vel_reward = self.config.reward.tracking_lin_vel * jp.exp(
            -jp.sum(jp.square(data.qvel[0:3] - state_info["base_vel_desire"][0:3]))
            / self.config.reward.tracking_sigma_vel
        )

        # control cost
        ctrl_cost = self.config.reward.ctrl_cost_weight * jp.sum(jp.square(data.qfrc_actuator[-self.model.nu :]))
        ctrl_cost = jp.clip(ctrl_cost, -1.0, 1.0)

        return {
            "ctrl_cost": ctrl_cost,
            "tracking_lin_vel_reward": tracking_lin_vel_reward,
            "tracking_ang_vel_reward": tracking_ang_vel_reward,
            "tracking_pos_reward": tracking_pos_reward,
            "tracking_orientation_reward": tracking_orientation_reward,
            "trtacking_joint_reward": tracking_joint_reward,
            "grf_penalty": grf_penalty,
        }

    def jt_blending(self, t: float, step_dur: float, action: jp.ndarray, state_info: Dict[str, Any]):
        """Blend the action with the offset."""
        offset = state_info["offset"]
        t0 = state_info["domain_info"]["step_start"]
        phaseVar = (t - t0) / step_dur
        behavstate = state_info["state"]
        dur = self._blend_dur[behavstate]
        lambda_y = jp.clip(phaseVar / dur, 0, 1)
        blended_action = (1 - lambda_y) * offset + action

        return blended_action

    def mechanical_power(self, data: mjx.Data) -> float:
        """Calculate the 2-norm sum of the mechanical power."""
        u = data.qfrc_actuator[-self.model.nu :]
        q_dot = data.qvel[-self.model.nu :]
        return jp.linalg.norm(u * q_dot, 2)

    def mcot(self, t_values, step_length, P_values):
        """Calculate the mechanical cost of transport."""
        # https://static1.squarespace.com/static/5c7b4b45a9ab95423630034b/t/6217aba70df6fb6a8f070dff/1645718443585/Algorithmic+Foundations....pdf

        # J = 1/mg*l * integral( P(u,q_dot) dt)
        g = 9.81  # acceleration due to gravity in m/s^2
        # Define the mechanical power function P as the sum of the 2-norm of torque and velocity products for each joint.

        # Compute the values of the integrand at the given time values
        # integrand_values = jp.array([P(data.qfrc_actuator[-model.nu:], data.qvel[-model.nu:]) for t in t_values])

        # Compute the approximate integral using the trapezoidal rule
        integral_value = trapezoid(P_values, t_values)

        # Calculate the cost function J
        return 1 / (self.mass * g * step_length) * integral_value

    def cop_reward(self, data: mjx.Data, state_info: Dict[str, Any]) -> float:
        """Calculate a reward based on the Center of Pressure (CoP).

        Args:
            cop (jax.Array): The current Center of Pressure.
            desired_cop_range (Tuple[float, float]): A tuple representing the desired range for the CoP.

        Returns:
            float: The calculated reward.
        """
        contact_force = self._get_contact_force(data)
        cop_L = self._calculate_cop(data.contact.pos[0:4], contact_force[0:4])
        cop_R = self._calculate_cop(data.contact.pos[4:8], contact_force[4:8])

        # Extract the desired min and max range for the CoP
        desired_min, desired_max = self.config.desired_cop_range

        step_start = state_info["domain_info"]["step_start"]
        phase_var = (data.time - step_start) / self.step_dur[state_info["state"]]
        # desired cop
        desired_cop = jp.zeros(2)
        desired_cop = desired_cop.at[0].set(phase_var * (desired_max - desired_min) + desired_min)

        # Calculate how far the CoP is from the desired range
        cop = cop_L[0:2]
        deviation_L = jp.square(cop - desired_cop)

        cop = cop_R[0:2]
        deviation_R = jp.square(cop - desired_cop)

        # Calculate the reward as the negative of the deviation
        # This means less deviation results in a higher (less negative) reward
        return self.config.reward.cop_scale * jp.sum(deviation_L + deviation_R)

    def _get_contact_force(self, data: mjx.Data) -> jax.Array:
        contact_force = data.efc_force[data.contact.efc_address]
        return contact_force

    def _calculate_cop(self, pos: jax.Array, force: jax.Array) -> jax.Array:
        def true_func(_):
            return jp.dot(pos.T, force) / jp.sum(force)

        def false_func(_):
            return jp.zeros_like(pos[0])

        # Use jax.lax.cond to check if sum of forces is not zero
        return jax.lax.cond(jp.sum(force) > 0, None, true_func, None, false_func)

    def _get_obs(self, data: mjx.Data, action: jp.ndarray, state_info: Dict[str, Any]) -> jp.ndarray:
        """Observes position, velocities, contact forces, and last action."""
        position = data.qpos
        velocity = data.qvel

        obs_list = []
        domain_idx = state_info["domain_info"]["domain_idx"]
        step_start = state_info["domain_info"]["step_start"]
        nominal_action = state_info["nominal_action"]
        phase_var = (data.time - step_start) / self.step_dur[state_info["state"]]

        # last action
        obs_list.append(position)  # 19
        obs_list.append(velocity)  # 18
        obs_list.append(self._get_contact_force(data))  # 8
        obs_list.append(nominal_action)  # 12
        obs_list.append(jp.array([domain_idx, step_start, phase_var]))

        obs = jp.clip(jp.concatenate(obs_list), -100.0, 2000.0)

        # stack observations through time
        single_obs_size = len(obs)
        state_info["obs_history"] = jp.roll(state_info["obs_history"], single_obs_size)
        state_info["obs_history"] = jp.array(state_info["obs_history"]).at[:single_obs_size].set(obs)

        return state_info["obs_history"]

    def getRender(self):
        """Get the renderer and camera for rendering."""
        camera = mj.MjvCamera()
        camera.azimuth = 0
        camera.elevation = 0
        camera.distance = 3
        camera.lookat = jp.array([0, 0, 0.5])
        self.camera = camera

        renderer = mj.Renderer(self.model, 480, 640)
        renderer._scene_option.flags[_enums.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        renderer._scene_option.sitegroup[2] = 0

        self.renderer = renderer
        return

    def get_image(self, state):
        """Get the image from the renderer."""
        d = mj.MjData(self.model)
        # write the mjx.Data into an mjData object
        mjx.device_get_into(d, state)
        mj.mj_forward(self.model, d)

        self.camera.lookat[0] = d.qpos[0]
        self.camera.lookat[1] = d.qpos[1]

        # use the mjData object to update the renderer

        self.renderer.update_scene(d, camera=self.camera)

        # time = d.time
        # curTime = f"Time = {time:.3f}"
        # mj.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,renderer._rect,curTime,'test',renderer._mjr_context)

        return self.renderer.render()

    def _ncr(self, n, r):
        r = min(r, n - r)
        numer = jp.prod(jp.arange(n, n - r, -1))
        denom = jp.prod(jp.arange(1, r + 1))
        return numer // denom

    def _forward(self, t, t0, step_dur, alpha):
        bez_deg = alpha.shape[1] - 1
        B = 0
        tau = (t - t0) / step_dur
        tau = jp.clip(tau, 0, 1)
        for i in range(bez_deg + 1):
            x = self._ncr(bez_deg, i)
            B = B + x * ((1 - tau) ** (bez_deg - i)) * (tau**i) * alpha[:, i]
        return B  # B.astype(jnp.float32)

    def _forward_vel(self, t, t0, step_dur, alpha):
        dB = 0
        tau = (t - t0) / step_dur
        tau = jp.clip(tau, 0, 1)
        for i in range(self.bez_deg):
            dB = dB + self.bez_deg * (alpha[:, i + 1] - alpha[:, i]) * self._ncr(self.bez_deg - 1, i) * (
                (1 - tau) ** (self.bez_deg - i - 1)
            ) * (tau**i)
        dtau = 1 / step_dur
        dB = dB * dtau
        return dB  # dB.astype(jnp.float32)

    def _remap_coeff(self):
        # assuming the last num_output is the only relevant ones
        baseRemap = jp.array([1, -1, 1, -1, 1, -1], dtype=jp.float32)
        legRemap = jp.array([-1, -1, 1, 1, 1, -1], dtype=jp.float32)
        relabel = jax.scipy.linalg.block_diag(jp.diag(baseRemap), jp.diag(legRemap), jp.diag(legRemap))
        relabelIdx = jp.array([0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11], dtype=jp.int32)
        R = jp.zeros_like(relabel)
        R = R.at[relabelIdx].set(relabel)
        return R
