import enum
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import mediapy as media
import mujoco as mj
import numpy as np
import yaml
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import PipelineEnv, State
from jax import lax
from jax.scipy.integrate import trapezoid
from mujoco import _enums, _structs, mjx
from mujoco.mjx._src import scan
from mujoco.mjx._src.types import DisableBit

from ambersim import ROOT
from ambersim.base import MjxEnv
from ambersim.utils.asset_utils import add_geom_to_env, add_heightfield_to_mujoco_xml, generate_boxes_xml
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


class StanceState(enum.IntEnum):
    """Stance states of the exoskeleton."""
    Left = 0
    Right = 1

@dataclass
class ExoRewardConfig:
    """Weightings for the reward function."""

    # Tracking rewards are computed using exp(-delta^2/sigma)
    # sigma can be a hyperparameters to tune.
    # Track the base x-y velocity (no z-velocity tracking.)
    tracking_lin_vel: float = 1.0
    # Track the angular velocity along z-axis, i.e. yaw rate.
    tracking_ang_vel: float = 1.0
    # Below are regularization terms, we roughly divide the
    # terms to base state regularizations, joint
    # regularizations, and other behavior regularizations.
    # Penalize the base velocity in z direction, L2 penalty.
    tracking_base_ori: float = 1.0
    # Penalize the base roll and pitch rate. L2 penalty.
    tracking_base_pos: float = 1.0
    tracking_sigma_pos: float = 1.0
    tracking_joint: float = 1.0
    tracking_sigma_joint_pos: float = 1.0
    tracking_sigma_vel: float = 1.0
    tracking_sigma_cop: float = 1.0
    unhealthy_penalty: float = -10.0
    healthy_reward: float = 1.0
    tracking_cop: float = 1.0

class ExoConfig:
    """config dataclass that specified the reward coefficient and other custom setting for exo env."""

    reward: ExoRewardConfig = ExoRewardConfig()
    terminate_when_unhealthy: bool = True
    reset_2_stand: bool = False
    reset_noise_scale: float = 1e-4
    history_size: float = 15 # TODO: history length; check update trajectory history update properly
    xml_file: str = "loadedExo.xml"
    jt_traj_file: str = "default_bez.yaml"
    loading_pos_file: str = "sim_config_loadingPos.yaml"
    ctrl_limit_file: str = "limits.yaml"
    position_ctrl: bool = True
    residual_action_space: bool = False
    physics_steps_per_control_step: int = 10
    action_scale: float = 0.05
    custom_action_space: jp.ndarray = jp.array([1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0])
    custom_act_idx: jp.ndarray = jp.array([0, 2, 6, 8])
    impact_threshold: float = 400.0
    impact_based_switching: bool = True
    no_noise: bool = True

    def to_dict(self):
        """Convert the dataclass to a dictionary."""
        return self.__dict__


class Exo(MjxEnv):
    """custom environment for the exoskeleton."""

    def __init__(self, config: ExoConfig = ExoConfig, **kwargs):
        """Initialize the environment."""
        self.config = config
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
        self.observation_size_single_step = self.model.nq + self.model.nv + 8 + 3 + 3
        self.curr_step = 0
        self.obs_history_update_freq = 1

        super().__init__(mj_model=self.model)  ##, physics_steps_per_control_step=self.config.physics_steps_per_control_step)
        # super().__init__(mj_model=self.model, physics_steps_per_control_step=self.config.physics_steps_per_control_step)

        self.efc_address = jp.array([0, 4, 8, 12, 16, 20, 24, 28]) # TODO: check mjx sensor feature; grf value reasonable?
        self.left_grf_idx = jp.array([0, 1, 2, 3])
        self.right_grf_idx = jp.array([4, 5, 6, 7])

    @property
    def action_size(self) -> int:
        """Override the super class action size function."""
        return self.custom_act_space_size

    def loadRelevantParams(self):
        """Load the relevant parameters for the environment."""
        self.base_frame_idx = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "torso")
        foot_geom_idx = [ # TODO: replace with site definition for sole
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "left_sole_frame"), # TODO: check mjOBJ_SITE, *value*
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "right_sole_frame"),
        ]
        self.foot_geom_idx = jp.array(foot_geom_idx)

        ankle_geom_idx = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "RightHenkeAnkleLink"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "LeftHenkeAnkleLink"),
        ]
        self.ankle_geom_idx = jp.array(ankle_geom_idx)

    def load_traj(self) -> None:
        """Load default trajectory from yaml file specfied in the config."""
        gait_params_file = os.path.join(ROOT, "..", "models", "exo", self.config.jt_traj_file)
        with open(gait_params_file, "r") as file:
            gait_params = yaml.safe_load(file)

        self.step_dur = 0.5 * jp.ones(9)
        self.step_dur = self.step_dur.at[BehavState.Walking].set(gait_params["step_dur"])

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

        self._q_default = self._q_default.at[BehavState.Walking.value, :].set(self._q_init)
        self._dq_default = self._dq_default.at[BehavState.Walking.value, :].set(self._dq_init)
        self.swing_foot = 0

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

    def reset(
        self,
        rng: jp.ndarray,
        q_init: jp.ndarray = None,
        dq_init: jp.ndarray = None,
        behavstate: BehavState = BehavState.Walking,
    ) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self.config.reset_noise_scale, self.config.reset_noise_scale

        qpos = self._q_default[behavstate, :]
        qvel = self._dq_default[behavstate, :]

        qpos_noise = jax.random.uniform(rng1, (self.sys.nq,), minval=0.1 * low, maxval=0.1 * hi)
        qvel_noise = self._dq_default[behavstate, :] + jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        if q_init is None:
            q_init = self._q_init
        if dq_init is None:
            dq_init = self._dq_init
        # if BehaveState is walking, then override the qpos and qvel with the desired values
        # using lax.cond to avoid jax error
        def true_fun(_):
            return q_init, dq_init

        def false_fun(_):
            return qpos, qvel

        qpos, qvel = jax.lax.cond(behavstate == BehavState.Walking, true_fun, false_fun, None)

        if not self.config.no_noise:
            qpos = qpos + qpos_noise
            qvel = qvel + qvel_noise

        data = self.pipeline_init(qpos, qvel)

        reward, done, zero = jp.zeros(3)

        state_info = {
            "state": behavstate,
            "offset": jp.zeros(12),
            "des_cop": data.geom_xpos[self.foot_geom_idx[0], 0:2],
            "domain_info": {
                "step_start": zero,
                "domain_idx": StanceState.Right.value,
                "update_geom": jp.ones(4) * -1.0,
            },
            "obs_history": jp.zeros(self.config.history_size * self.observation_size_single_step),
            "nominal_action": jp.zeros(12),
            "blended_action": jp.zeros(12),
            "base_pos_desire": jp.zeros(6),
            "base_vel_desire": jp.zeros(6),
            "joint_desire": jp.zeros(12),
            "last_action": jp.zeros(self.custom_act_space_size),
            "tracking_err": jp.zeros(12),
            "com_pos": jp.zeros(3),
            "com_vel": jp.zeros(3),
            "reward_tuple": {
                "tracking_lin_vel_reward": zero,
                "tracking_ang_vel_reward": zero,
                "tracking_pos_reward": zero,
                "tracking_orientation_reward": zero,
                "tracking_joint_reward": zero,
                "tracking_foot_reward": zero,
                "cop_reward": zero,
            },
            "alpha": self.alpha,
            "alpha_base": self.alpha_base,
            "foot_target": {
                "left": jp.zeros(3),
                "right": jp.zeros(3),
                "tracking_foot_reward_cached": 0.0,
            },
            "impact_threshold": self.config.impact_threshold,
            "contact_force": zero,
            "debug": {

            },
            "accelerations" : jp.zeros(18),
            "contact_forces" : jp.zeros(8),
            "f_grad" : jp.zeros(self.config.history_size),
        }
        """
        "foot_target": jp.zeros(6),
        """

        self.curr_step = 0
        obs = self._get_obs(data, jp.zeros(self.action_size), state_info)

        metrics = {"total_dist": 0.0}
        for k in state_info["reward_tuple"]:
            metrics[k] = state_info["reward_tuple"][k]

        return State(data, obs, reward, done, metrics, state_info)

    def getFootPos(self, state: State) -> jp.ndarray:
        """Get the foot position in world frame."""
        data = state.pipeline_state
        foot_pos = data.site_xpos[self.foot_geom_idx, 0:3] # TODO: check geom_xpos, site_xpos
        return foot_pos

    def checkDone(self, data: mjx.Data) -> float:
        """Check if the robot is falling or joint limits are reached."""
        # resetting logic if joint limits are reached or robot is falling
        # violate joint limit
        joint_angles = data.qpos[-self.model.nu :]
        done = 0.0
        done = jp.where(
            jp.logical_or(
                jp.any(joint_angles < self._jt_lb + 0.01),
                jp.any(joint_angles > self._jt_ub - 0.01),
            ),
            1.0,
            done,
        )
        # no ground reaction force
        done = jp.where(jp.sum(self._get_contact_force(data)) < 10.0, 1.0, done)

        # falling
        done = jp.where(data.qpos[2] < 0.7, 1.0, done)
        # jax.debug.print("Z pos: {}", data.qpos[2])
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
        des_cop = state.info["des_cop"]

        # action = self.conv_action_based_on_idx(cur_action, jp.zeros(12))
        def update_step(step_start, domain_idx):
            new_step_start = data0.time
            new_domain_idx = 1 - domain_idx  # Switch domain_idx between 0 and 1
            # if current time - step_Start <

            return new_step_start, new_domain_idx

        def no_update(step_start, domain_idx):
            return step_start, domain_idx

        condition = (data0.time - step_start) / self.step_dur[BehavState.Walking] >= 1
        if self.config.impact_based_switching:
            condition = jp.logical_and(
                self.checkImpact(data0, state.info),
                (data0.time - step_start) / self.step_dur[BehavState.Walking] >= 0.8,
            )
        
        def step_switch_target():
            return self.switchFootTarget(state, data0)
        state.info["foot_target"]["left"], state.info["foot_target"]["right"] = lax.cond(condition, step_switch_target, lambda : (state.info["foot_target"]["left"], state.info["foot_target"]["right"]))

        new_step_start, domain_idx = lax.cond(
            condition,
            lambda args: update_step(*args),
            lambda args: no_update(*args),
            (step_start, domain_idx),
        )
        step_start = new_step_start
        alpha = lax.cond(
            domain_idx == StanceState.Right.value,
            lambda _: state.info["alpha"],
            lambda _: jp.dot(self.R, state.info["alpha"]),
            None,
        )
        alpha_base = lax.cond(
            domain_idx == 0,
            lambda _: state.info["alpha_base"],
            lambda _: jp.dot(self.R_base, state.info["alpha_base"]),
            None,
        )
        q_desire = self._forward(data0.time, step_start, self.step_dur[BehavState.Walking], alpha)

        q_actual = data0.qpos[-self.model.nu :]
        new_offset = lax.cond(condition, lambda _: q_actual - q_desire, lambda _: state.info["offset"], None)

        state.info["nominal_action"] = q_desire
        state.info["base_pos_desire"] = self._forward(
            data0.time, step_start, self.step_dur[BehavState.Walking], alpha_base
        )
        state.info["base_vel_desire"] = self._forward_vel(
            data0.time, step_start, self.step_dur[BehavState.Walking], alpha_base
        )
        state.info["domain_info"]["domain_idx"] = domain_idx
        state.info["domain_info"]["step_start"] = step_start
        state.info["offset"] = new_offset

        return q_desire, state

    def step_wrapper(self, state_vec: jp.ndarray, action: jp.ndarray):
        return ...

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        cur_action = action
        data0 = state.pipeline_state

        q_desire, state = self.getNominalDesire(state)
        scaled_action = self.config.action_scale * action
        if self.config.residual_action_space:
            action = q_desire.at[self.config.custom_act_idx].set(q_desire[self.config.custom_act_idx] + scaled_action)
        else:
            action = q_desire + scaled_action
        state.info["joint_desire"] = action

        blended_action = self.jt_blending(data0.time, self.step_dur[state.info["state"]], action, state.info)
        motor_targets = jp.clip(blended_action, self._jt_lb, self._jt_ub)
        data = self.pipeline_step(data0, motor_targets)
        self.curr_step += 1 # TODO: check curr_step getting updated; else insert into state_info
        # observation data
        obs = self._get_obs(data, action, state.info)
        
        reward_tuple = self.evaluate_reward(data0, data, cur_action, state.info)

        # self.record_state(state, {"contact_force": self._get_contact_force(data),
        #                           "3_foot_pos": {"foot_pos_L": data.site_xpos[self.foot_geom_idx][0],
        #                                          "foot_pos_R": data.site_xpos[self.foot_geom_idx][1]},
        #                           "3_foot_tgt": {"foot_tgt_L": state.info["foot_target"]["left"],
        #                                          "foot_tgt_R": state.info["foot_target"]["right"]},
        #                           "3_cop": {"cop": state.info["debug"]["cop"]},
        #                           })
        # state management

        state.info["reward_tuple"] = reward_tuple
        state.info["last_action"] = cur_action
        state.info["blended_action"] = blended_action
        state.info["tracking_err"] = (
            data.qpos[-self.model.nu :] - action
        )  # currently assuming motor_targets is the desired joint angles; TODO handle torque case
        state.info["com_pos"] = data.subtree_com[1]

        ## AHAC ##
        state.info["accelerations"] = data.efc_aref[self.efc_address]
        state.info["contact_forces"] = self._get_contact_force(data) #[data.cfrc_ext[i] for i in self.foot_geom_idx]
        #########

        # resetting logic if joint limits are reached or robot is falling
        # violate joint limit
        done = self.checkDone(data)
        reward = jp.sum(jp.array(list(reward_tuple.values())))
        reward = reward + (1 - done) * self.config.reward.healthy_reward + done * self.config.reward.unhealthy_penalty
        step_condense = lambda state_vec, action : reward
        f_grad_vec = jax.grad(step_condense)(state.info["obs_history"],action)
        obs_size=self.observation_size_single_step
        f_grad = jp.array([jp.linalg.norm(f_grad_vec[i*obs_size:(i+1)*obs_size-1]) for i in range(self.config.history_size)])
        state.info["f_grad"] = f_grad

        # Update metrics
        for k in state.info["reward_tuple"]:
            state.metrics[k] = state.info["reward_tuple"][k]
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)
    
    def state_condition_met(self, maxErrorTrigger, minTransitionTime, data: mjx.Data, state_info: Dict[str, Any]):
        """Check if the error is within the threshold and minimum time has passed since the last transition."""
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
        num_dims = len(values[0]) if isinstance(values[0], list or jp.ndarray) else 1
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
        # print("logged_data: ", logged_data)
        values_d = {}
        values = []
        for attribute in attributes:
            for data_point in logged_data:
                print(attribute, data_point[attribute])
                if attribute in data_point and not isinstance(data_point[attribute], dict):
                    values = [data_point[attribute] for data_point in logged_data]
                else:
                    for key in data_point[attribute].keys():
                        if f"{attribute}:{key}" not in values_d.keys():
                            values_d[f"{attribute}:{key}"] = []
                        values_d[f"{attribute}:{key}"].append(data_point[attribute][key])
            if values or values_d:
                if attribute[-1] == "3":
                    self.plot_3d_attribute(attribute, values_d, save_dir)
                else:
                    self.plot_attribute(attribute, values, save_dir)
            else:
                print(f"{attribute} is not valid")

    def load_and_plot_data(self, file_path, plot_save_dir):
        """Loads data from a pickle file and plots the logged data.

        Args:file_path (str): Path to the pickle file containing logged data.
            plot_save_dir (str): Directory where plots should be saved.
        """
        if not os.path.exists(file_path):
            print(f"File {file_path} not found.")
            return

        with open(file_path, "rb") as file:
            logged_data = pickle.load(file)

        self.plot_logged_data(logged_data, plot_save_dir)

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

    def eulRates2omega(self, eulRates, orientation):
        """Convert euler rates to omega."""
        # Input processing
        assert len(eulRates) == 3, "The omega must be a numerical vector of length 3."

        x, y, z = orientation

        M = jp.array([[jp.cos(y) * jp.cos(z), jp.sin(z), 0], [-jp.cos(y) * jp.sin(z), jp.cos(z), 0], [jp.sin(y), 0, 1]])

        omega = jp.dot(M, eulRates)

        return omega

    def eulerXYZ2quat(self, eul):
        """Convert Euler angles (XYZ order) to a quaternion.

        Args:
            eul (array): Euler angles, a 3-element array-like object.

        Returns:
            array: Quaternion as a 4-element array.
        """
        c = jp.array([jp.cos(eul[0] / 2), jp.cos(eul[1] / 2), jp.cos(eul[2] / 2)])
        s = jp.array([jp.sin(eul[0] / 2), jp.sin(eul[1] / 2), jp.sin(eul[2] / 2)])

        q = jp.array(
            [   c[0] * c[1] * c[2] - s[0] * s[1] * s[2],
                s[0] * c[1] * c[2] + c[0] * s[1] * s[2],
                -s[0] * c[1] * s[2] + c[0] * s[1] * c[2],
                c[0] * c[1] * s[2] + s[0] * s[1] * c[2],
            ])

        return q

    def quat2eulXYZ(self, quat):
        """Convert quaternion to euler angles."""
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
            [   jp.arctan2(-2.0 * (z1_idx_2 * b - z1_idx_1 * z1_idx_0), ((eul_tmp - b_eul_tmp) - c_eul_tmp) + d_eul_tmp),
                jp.arcsin(aSinInput),
                jp.arctan2(-2.0 * (z1_idx_1 * z1_idx_2 - b * z1_idx_0), ((eul_tmp + b_eul_tmp) - c_eul_tmp) - d_eul_tmp),
            ])

        return eul

    def evaluate_reward(self, data0: mjx.Data, data: mjx.Data, action: jp.ndarray, state_info: Dict[str, Any]) -> dict:
        """Evaluate the reward function."""
        # convert relevant fields
        eul = self.quat2eulXYZ(data.qpos[3:7])
        eul_rate = self.eulRates2omega(data.qvel[3:6], eul)

        # base coordinate tracking
        domain_idx = state_info["domain_info"]["domain_idx"]

        step_start = state_info["domain_info"]["step_start"]
        phase_var = (data.time - step_start) / self.step_dur[state_info["state"]]

        def get_base_pos(domain_idx, data, foot_geom_idx):
            # base_pos = data.qpos[0:3] - data.geom_xpos[foot_geom_idx[domain_idx], 0:3]
            base_pos = data.subtree_com[1] - data.site_xpos[self.foot_geom_idx[domain_idx], 0:3] # geom -> site
            return base_pos

        base_pos = jax.lax.cond(
            domain_idx == 0,
            lambda _: get_base_pos(0, data, self.foot_geom_idx),
            lambda _: get_base_pos(1, data, self.foot_geom_idx),
            operand=None,
        )

        tracking_pos_reward = self.config.reward.tracking_base_pos * jp.exp(
            -jp.sum(jp.square(base_pos[0:2] - state_info["base_pos_desire"][0:2])) / self.config.reward.tracking_sigma_pos
        )
        
        tracking_orientation_reward = self.config.reward.tracking_base_ori * jp.exp(
            -jp.sum(jp.square(eul - state_info["base_pos_desire"][3:6])) / self.config.reward.tracking_sigma_pos
        )

        tracking_joint_reward_raw = self.config.reward.tracking_joint * jp.exp(
            -jp.sum(jp.square(data.qpos[-self.model.nu :] - state_info["nominal_action"]))
            / self.config.reward.tracking_sigma_joint_pos
        )

        # phase_func_val = 
        # tracking_joint_reward = jax.lax.cond(jp.logical_or(phase_var < 0.1, phase_var > 0.9),
        #                                     lambda: 0.0,
        #                                     lambda: tracking_joint_reward_raw)
        tracking_joint_reward = tracking_joint_reward_raw * 0.001

        tracking_ang_vel_reward = self.config.reward.tracking_ang_vel * jp.exp(
            -jp.sum(jp.square(eul_rate - state_info["base_vel_desire"][3:6])) / self.config.reward.tracking_sigma_vel
        )

        tracking_lin_vel_reward = self.config.reward.tracking_lin_vel * jp.exp(
            -jp.sum(
                jp.square(
                    math.rotate(data.qvel[:3], math.quat_inv(data.qpos[3:7])) - state_info["base_vel_desire"][0:3]
                )
            )
            / self.config.reward.tracking_sigma_vel
        )

        currentFootPos = data.site_xpos[self.foot_geom_idx] # TODO: plot foot pos and targets
        d = -(jp.linalg.norm(currentFootPos[0,:2] - state_info["foot_target"]["left"][:2]) +
              jp.linalg.norm(currentFootPos[1,:2] - state_info["foot_target"]["right"][:2]))

        tracking_foot_reward = jax.lax.cond(phase_var < 0.8,
                                            lambda: state_info["foot_target"]["tracking_foot_reward_cached"],
                                            lambda: (1-jp.exp(1/d)))

        state_info["foot_target"]["tracking_foot_reward_cached"] = tracking_foot_reward

        cop_reward = self.cop_reward(data, state_info)

        return {
            "tracking_lin_vel_reward": self._clip_reward(tracking_lin_vel_reward),
            "tracking_ang_vel_reward": self._clip_reward(tracking_ang_vel_reward) ,
            "tracking_pos_reward": self._clip_reward(tracking_pos_reward),
            "tracking_orientation_reward": self._clip_reward(tracking_orientation_reward),
            "tracking_joint_reward": self._clip_reward(tracking_joint_reward),
            "tracking_foot_reward": self._clip_reward(tracking_foot_reward),
            "cop_reward": self._clip_reward(cop_reward),
        }

    def get_swing_grf(self, data: mjx.Data, state_info) -> jax.Array: # TODO: check consistency func calls
        """Get the ground reaction force for the swing leg."""
        domain_idx = state_info["domain_info"]["domain_idx"]

        def get_swing_contact(idx, data):
            """Get the contact force for the stance leg."""
            contact_force = self._get_contact_force(data)
            return contact_force[idx]

        swing_grf = jax.lax.cond(
            domain_idx == StanceState.Right.value,
            lambda _: get_swing_contact(self.left_grf_idx, data),
            lambda _: get_swing_contact(self.right_grf_idx, data),
            operand=None,
        )

        return swing_grf

    def _clip_reward(self, reward: float) -> float:
        """Clip the reward."""
        return jp.clip(reward, -50.0, 50.0)

    def _reward_tracking_lin_vel(self, commands: jax.Array, x: Transform, xd: Motion) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jp.exp(-lin_vel_error / self.config.reward.tracking_sigma)
        return lin_vel_reward

    def _reward_tracking_ang_vel(self, commands: jax.Array, x: Transform, xd: Motion) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        return jp.exp(-ang_vel_error / self.config.reward.tracking_sigma)

    def checkImpact(self, data, state_info):
        """Check if the robot is in impact."""
        swing_grf = self.get_swing_grf(data, state_info)
        return jp.sum(swing_grf) > state_info["impact_threshold"]

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

    def cop_reward(self, data: mjx.Data, state_info: Dict[str, Any]) -> float:
        """Calculate a reward based on the Center of Pressure (CoP).

        Args:
            cop (jax.Array): The current Center of Pressure.
            desired_cop_range (Tuple[float, float]): A tuple representing the desired range for the CoP.

        Returns:
            float: The calculated reward.
        """
        contact_force = self._get_contact_force(data)

        domain_idx = state_info["domain_info"]["domain_idx"]

        step_start = state_info["domain_info"]["step_start"]
        phase_var = (data.time - step_start) / self.step_dur[state_info["state"]]

        # desired cop
        desired_cop = jp.zeros(2)
        # desired_cop = desired_cop.at[0].set(phase_var * (desired_max - desired_min) + desired_min)

        stance_grf_idx = jax.lax.cond(
            domain_idx == StanceState.Right.value,
            lambda _: self.right_grf_idx,
            lambda _: self.left_grf_idx,
            operand=None,
        )

        cop = self._calculate_cop(data.contact.pos[stance_grf_idx], contact_force[stance_grf_idx]) # TODO: plot cops
        # state_info["debug"]["cop"] = cop
        sole_pos = data.site_xpos[self.foot_geom_idx[domain_idx]]

        desired_cop = sole_pos[0:2]
        # desired_cop = desired_cop.at[1].set(0)

        # Calculate the reward as the negative of the deviation
        # This means less deviation results in a higher (less negative) reward
        return self.config.reward.tracking_cop * jp.exp(
            -jp.sum(jp.square(cop[0:2] - desired_cop)) / self.config.reward.tracking_sigma_cop
        )

    def _get_contact_force(self, data: mjx.Data) -> jax.Array: # TODO: check contact force; support for contact; plot
        # hacked version
        # import ipdb; ipdb.set_trace()
        contact_force = data.efc_force[self.efc_address]

        # mu = data.contact.friction
        dim = 3  # hardcode to pyramid

        # TODO: get rid of the for loop
        for i in range(len(self.efc_address)):
            # decode pyramid
            force_normal = 0.0
            for j in range(2 * (dim - 1)):
                force_normal = force_normal + data.efc_force[self.efc_address[i] + j]

            contact_force = contact_force.at[i].set(force_normal)

        return contact_force

    def _calculate_cop(self, pos: jax.Array, force: jax.Array) -> jax.Array:
        def true_func(_):
            return jp.dot(pos.T, force) / jp.sum(force)

        def false_func(_):
            return jp.zeros_like(pos[0])

        # Use jax.lax.cond to check if sum of forces is not zero
        return jax.lax.cond(jp.any(force) > 10.0, None, true_func, None, false_func)

    def _get_obs(self, data: mjx.Data, action: jp.ndarray, state_info: Dict[str, Any]) -> jp.ndarray:
        """Observes position, velocities, contact forces, and last action."""
        position = data.qpos
        velocity = data.qvel

        obs_list = []
        domain_idx = state_info["domain_info"]["domain_idx"]
        step_start = state_info["domain_info"]["step_start"]
        nominal_base_desire = state_info["base_vel_desire"][0:3]
        phase_var = (data.time - step_start) / self.step_dur[state_info["state"]]
        contact_force = self._get_contact_force(data)

        # last action
        obs_list.append(position)  # 19
        obs_list.append(velocity)  # 18
        obs_list.append(contact_force)  # 8
        obs_list.append(nominal_base_desire)  # 3
        obs_list.append(jp.array([domain_idx, step_start, phase_var]))

        obs = jp.clip(jp.concatenate(obs_list), -100.0, 2000.0)
        # self.obs_buffer

        # stack observations through time
        if self.curr_step % self.obs_history_update_freq == 0 and self.curr_step != 0:
            single_obs_size = len(obs)
            state_info["obs_history"] = jp.roll(
                state_info["obs_history"], (single_obs_size * self.obs_history_update_freq)
            )
            # temporary
            # state_info["obs_history"] = (
            #     jp.array(state_info["obs_history"]).at[: (single_obs_size * self.obs_history_update_freq)].set(obs)
            # )
            state_info["obs_history"] = (
                jp.array(state_info["obs_history"]).at[: (single_obs_size * self.obs_history_update_freq)].set(obs)
            ) # TODO: confirm correctness 

        return state_info["obs_history"]


    def getRender(self):
        """Get the renderer and camera for rendering."""
        camera = mj.MjvCamera()
        camera.azimuth = 60
        camera.elevation = 0.6
        camera.distance = 3
        camera.lookat = jp.array([0, 0, 0.8])
        self.camera = camera

        renderer = mj.Renderer(self.model, 480, 640)
        # renderer = mj.Renderer(self.model)
        renderer._scene_option.flags[_enums.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        renderer._scene_option.sitegroup[2] = 0

        self.renderer = renderer
        return

    def get_image(self, state):
        """Get the image from the renderer."""
        # write the mjx.Data into an mjData object
        d = mjx.get_data(self.model, state)
        mj.mj_forward(self.model, d)
        # use the mjData object to update the renderer
        self.renderer.update_scene(d, camera=self.camera)

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

        return B  # B.astype(jp.float32)

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
        return dB  # dB.astype(jp.float32)

    def _remap_coeff(self):
        # assuming the last num_output is the only relevant ones
        baseRemap = jp.array([1, -1, 1, -1, 1, -1], dtype=jp.float32)
        legRemap = jp.array([-1, -1, 1, 1, 1, -1], dtype=jp.float32)
        relabel = jax.scipy.linalg.block_diag(jp.diag(baseRemap), jp.diag(legRemap), jp.diag(legRemap))
        relabelIdx = jp.array([0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11], dtype=jp.int32)
        R = jp.zeros_like(relabel)
        R = R.at[relabelIdx].set(relabel)
        return R
    
    def switchFootTarget(self, state, data):
        # use state info, and key from run
        prng_key = jax.random.PRNGKey(seed=10)
        step_len = [0.11966493318671938, -0.2940499740759257, -7.344087977321223E-4]
        domain_idx = state.info["domain_info"]["domain_idx"]
        footPos = self.getFootPos(state)
        stanceFootPos = footPos[(domain_idx + 1) % 2]
        swingFootPos = footPos[domain_idx]
        targetOffset = jp.array([step_len[0] * 2, 0.0, 0.0])
        randomization = jp.array([jax.random.uniform(prng_key, minval=-0.05, maxval=0.05), 
                                  jax.random.uniform(prng_key, minval=-0.02, maxval=0.02), 0.0])

        def leftFootSwing():
            leftfootTarget = jp.add(swingFootPos, targetOffset)
            leftfootTarget = jp.add(leftfootTarget, randomization)
            rightfootTarget = stanceFootPos
            return leftfootTarget, rightfootTarget

        def rightFootSwing():
            leftfootTarget = stanceFootPos
            rightfootTarget = jp.add(swingFootPos, targetOffset)
            rightfootTarget = jp.add(rightfootTarget, randomization)
            return leftfootTarget, rightfootTarget

        leftfootTarget, rightfootTarget = lax.cond(domain_idx == StanceState.Right.value, leftFootSwing, rightFootSwing)

        return leftfootTarget, rightfootTarget


    # def record_state(self, state, quant_dict):
    #     for key in quant_dict.keys():
            # state.info["debug"][key] = quant_dict[key]
    
    from mpl_toolkits.mplot3d import Axes3D

    def plot_3d_attribute(self, attribute, vector_dict, save_dir):
        print(attribute, vector_dict)
        # Create a new figure for each plot
        for title, vectors in vector_dict.items():
            # Create a new figure for each plot title
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Loop through each vector and plot it
            for vector in vectors:
                # Unpack the vector components
                x, y, z = vector
                # Plot the vector from the origin to the point (x, y, z)
                ax.scatter(x, y, z)

            # Set the title of the plot
            ax.set_title(attribute)
            # ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
            plt.savefig(os.path.join(save_dir, f"{title}.png"))

