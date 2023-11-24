import os
from dataclasses import dataclass
from typing import Any, Dict

import jax
import jax.numpy as jp
import mujoco as mj
import numpy as np
import yaml
from brax.base import Base, Motion, Transform
from brax.envs.base import PipelineEnv, State
from jax import lax
from mujoco import _enums, _structs, mjx
from mujoco.mjx._src import scan
from mujoco.mjx._src.types import DisableBit

from ambersim import ROOT
from ambersim.base import MjxEnv
from ambersim.utils.io_utils import set_actuators_type


@dataclass
class ExoRewardConfig:
    """Weightings for the reward function."""

    # Tracking rewards are computed using exp(-delta^2/sigma)
    # sigma can be a hyperparameters to tune.
    # Track the base x-y velocity (no z-velocity tracking.)
    tracking_lin_vel: float = 2

    # Track the angular velocity along z-axis, i.e. yaw rate.
    tracking_ang_vel: float = 0.5

    # Below are regularization terms, we roughly divide the
    # terms to base state regularizations, joint
    # regularizations, and other behavior regularizations.
    # Penalize the base velocity in z direction, L2 penalty.
    lin_vel_z: float = -2.0

    # Penalize the base roll and pitch rate. L2 penalty.
    ang_vel_xy: float = -0.01

    # Penalize non-zero roll and pitch angles. L2 penalty.
    # orientation: float = -5.0

    # L2 regularization of joint torques, |tau|^2.
    ctrl_cost_weight: float = -0.002

    # Penalize the change in the action and encourage smooth
    # actions. L2 regularization |action - last_action|^2
    action_rate: float = -1.0

    # Penalize torque
    ctrl_cost_weight: float = -0.00001

    # Early termination penalty.
    termination: float = -1.0

    # Penalizing foot slipping on the ground.
    foot_slip: float = -0.1

    # Tracking reward = exp(-error^2/sigma).
    tracking_sigma: float = 0.25

    # forward reward
    forward_reward_weight: float = 0.5

    # healthy reward
    healthy_reward: float = 1.0

    # cop penalty
    cop_scale: float = -0.5


class ExoConfig:
    """config dataclass that specified the reward coefficient and other custom setting for exo env."""

    reward: ExoRewardConfig = ExoRewardConfig()
    terminate_when_unhealthy: bool = True
    reset_2_stand: bool = False
    healthy_z_range: tuple = (0.85, 1)
    desired_cop_range: tuple = (-0.1, 0.1)
    reset_noise_scale: float = 1e-2
    history_size: float = 5
    xml_file: str = "loadedExo.xml"
    jt_traj_file: str = "jt_bez_2023-09-10.yaml"
    loading_pos_file: str = "sim_config_loadingPos.yaml"
    ctrl_limit_file: str = "limits.yaml"
    rand_terrain: bool = False
    position_ctrl: bool = True
    residual_action_space: bool = True
    physics_steps_per_control_step: int = 10
    action_scale: jp.ndarray = jp.array([0.001, 0.001, 0.001, 0.001, 0.05, 0.1, 0.001, 0.001, 0.001, 0.001, 0.05, 0.1])
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
        self.observation_size_single_step = (
            self.model.nq - 3 + self.model.nv - 3 + 8 + 6 + self.custom_act_space_size + 3
        )

        super().__init__(mj_model=self.model, physics_steps_per_control_step=self.config.physics_steps_per_control_step)

    @property
    def action_size(self) -> int:
        """Override the super class action size function."""
        return self.custom_act_space_size

    def loadRelevantParams(self):
        """Load the relevant parameters for the environment."""
        self.base_frame_idx = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "torso")
        foot_geom_idx = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "left_sole"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "right_sole"),
        ]
        self.foot_geom_idx = jp.array(foot_geom_idx)

    def load_traj(self) -> None:
        """Load default trajectory from yaml file specfied in the config."""
        gait_params_file = os.path.join(ROOT, "..", "models", "exo", self.config.jt_traj_file)
        with open(gait_params_file, "r") as file:
            gait_params = yaml.safe_load(file)

        self.step_dur = gait_params["step_dur"]
        self.step_start = 0.0
        coeff_jt = np.reshape(np.array(gait_params["coeff_jt"]), (12, 8), order="F")
        self.alpha = jp.array(coeff_jt)
        self.bez_deg = 7
        self._q_init = jp.concatenate([jp.array(gait_params["ffPos"]), jp.array(gait_params["startingPos"])], axis=0)
        self._dq_init = jp.concatenate([jp.array(gait_params["ffVel"]), jp.array(gait_params["startingVel"])], axis=0)

        R = self._remap_coeff()
        self.R = R[-self.model.nu :, -self.model.nu :]
        self.q_desire = self._q_init[-self.model.nu :]
        loading_pos_file = os.path.join(ROOT, "..", "models", "exo", self.config.loading_pos_file)
        with open(loading_pos_file, "r") as file:
            load_params = yaml.safe_load(file)

        self._q_load = jp.concatenate([jp.array(load_params["ffPos"]), jp.array(load_params["startingPos"])], axis=0)

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

        if self.config.position_ctrl:
            for jt_idx in range(self.model.nu):
                self.model = set_actuators_type(self.model, "position", jt_idx, kp=self._p_gains[jt_idx])

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self.config.reset_noise_scale, self.config.reset_noise_scale

        if self.config.reset_2_stand:
            qpos = self._q_load + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
            qvel = jp.zeros(self.sys.nv)
        else:
            qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)
            qvel = self._dq_init + qvel
            qpos = self._q_init + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        # hardcode new_cmd for now
        new_cmd = jp.array([0.15, 0, 0])
        reward, done, zero = jp.zeros(3)

        state_info = {
            "domain_info": {
                "step_start": 0.0,
                "domain_idx": 0.0,
            },
            "rng": rng,
            "last_full_act": jp.zeros(self.model.nu),
            "last_act": jp.zeros(self.action_size),
            "last_vel": jp.zeros(12),
            # "last_contact_buffer": jp.zeros((20, 4), dtype=bool),
            "command": new_cmd,
            # "last_contact": jp.zeros(4, dtype=bool),
            # "feet_air_time": jp.zeros(4),
            "obs_history": jp.zeros(self.config.history_size * self.observation_size_single_step),
            "nominal_action": jp.zeros(12),
            "reward_tuple": {
                "forward_reward": zero,
                "healthy_reward": zero,
                "ctrl_cost": zero,
                "tracking_lin_vel_reward": zero,
                "tracking_ang_vel_reward": zero,
                "lin_vel_z_penalty": zero,
                "ang_vel_xy_penalty": zero,
                "cop_penalty": zero,
                "action_rate_penalty": zero,
                "total_reward": zero,
            },
            "step": 0,
        }

        obs = self._get_obs(data, jp.zeros(self.action_size), state_info)

        metrics = {"total_dist": 0.0}
        for k in state_info["reward_tuple"]:
            metrics[k] = state_info["reward_tuple"][k]

        return State(data, obs, reward, done, metrics, state_info)

    def reset_bez(self, rng: jp.ndarray, alpha: jp.ndarray, step_dur: float) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        # low, hi = -self.config.reset_noise_scale, self.config.reset_noise_scale

        q_init = self._q_init.at[-self.model.nu :].set(self._forward(self.step_start, self.step_start, step_dur, alpha))
        dq_init = self._dq_init.at[-self.model.nu :].set(
            self._forward_vel(self.step_start, self.step_start, step_dur, alpha)
        )

        # qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)
        qvel = dq_init
        # qvel = jp.zeros(self.sys.nv)
        qpos = q_init
        # qpos = qpos.at[-self.model.nu:] jax.random.uniform(rng1, (self.sys.nu,), minval=low, maxval=hi)
        # qpos = qpos.at[2].set(qpos[2]+10)
        data = self.pipeline_init(qpos, qvel)

        # hardcode new_cmd for now
        new_cmd = jp.array([0.15, 0, 0])
        reward, done, zero = jp.zeros(3)

        state_info = {
            "domain_info": {
                "step_start": 0.0,
                "domain_idx": 0.0,
            },
            "rng": rng,
            "last_full_act": jp.zeros(self.model.nu),
            "last_act": jp.zeros(self.action_size),
            "last_vel": jp.zeros(12),
            "command": new_cmd,
            "mcot": zero,
            "obs_history": jp.zeros(self.config.history_size * self.observation_size_single_step),
            "nominal_action": jp.zeros(12),
            "reward_tuple": {
                "forward_reward": zero,
                "healthy_reward": zero,
                "ctrl_cost": zero,
                "tracking_lin_vel_reward": zero,
                "tracking_ang_vel_reward": zero,
                "lin_vel_z_penalty": zero,
                "ang_vel_xy_penalty": zero,
                "cop_penalty": zero,
                "action_rate_penalty": zero,
                "total_reward": zero,
            },
            "step": 0,
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

    def convert_action_to_full_dimension(self, reduced_action):
        """Convert an action from the reduced action space to the full dimension action space.

        Args:
            reduced_action (jax.numpy.ndarray): The action in the reduced action space.

        Returns:
            jax.numpy.ndarray: The action in the full dimension action space.
        """
        action_mask = self.config.custom_action_space

        # Initialize a full dimension action array
        full_dimension_action = jp.zeros_like(action_mask, dtype=jp.float32)

        # Keep track of the index in the reduced action
        reduced_idx = 0

        # Function to handle true condition
        def true_fun(_):
            nonlocal reduced_idx
            value = reduced_action[reduced_idx]
            reduced_idx += 1
            return value

        # Function to handle false condition
        def false_fun(_):
            return 0.0

        # Iterate over the mask
        for i in range(action_mask.shape[0]):
            # Using lax.cond for conditional assignment
            full_dimension_action = full_dimension_action.at[i].set(
                self.config.action_scale[i] * lax.cond(action_mask[i] == 1, true_fun, false_fun, None)
            )

        return full_dimension_action

    def conv_action_based_on_idx(self, modified_action: jp.ndarray, nominal_action: jp.ndarray):
        """Convert an action from the reduced action space to the full dimension action space.

        Args:
            reduced_action (jax.numpy.ndarray): The action in the reduced action space.

        Returns:
            jax.numpy.ndarray: The action in the full dimension action space.
        """
        nominal_action = nominal_action.at[self.config.custom_act_idx].set(modified_action[self.config.custom_act_idx])

        return nominal_action

    def bez_step(self, state: State, alpha: jp.ndarray, step_dur: float) -> State:
        """Runs one timestep of the environment's dynamics."""
        rng, rng_noise, cmd_rng = jax.random.split(state.info["rng"], 3)

        data0 = state.pipeline_state
        step_start = state.info["domain_info"]["step_start"]
        action = self._forward(data0.time, step_start, step_dur, alpha)

        if self.config.position_ctrl:
            motor_targets = jp.clip(action, self._jt_lb, self._jt_ub)
        else:
            motor_targets = jp.clip(action, self._torque_lb, self._torque_ub)

        data = self.pipeline_step(data0, motor_targets)

        # observation data
        mg = 9.81 * self.mass
        vhip = data.qvel[0]
        u = data.qfrc_actuator[-self.model.nu :]
        dx = data.qvel[-self.model.nu :]
        mcot = jp.sqrt(jp.sum(jp.square(u * dx))) / (mg * jp.abs(vhip))

        # # calculate power
        # power = np.dot(u, dx)

        # # calculate mechanical work done
        # work = power / mg

        # # calculate distance traveled per unit time
        # distance_per_time = np.linalg.norm(dx) / dt

        # # calculate cost of transport
        # if distance_per_time == 0:
        #     cost_of_transport=0
        # else:
        #     cost_of_transport=work / distance_per_time

        jax.debug.print("mcot: {}", mcot)

        # vhip = dx('BasePosX');
        # mg   = 9.81 * sum([behavior.robotModel.Links(:).Mass]);
        # u  = domain.Inputs.Control.u;
        # Be = domain.Gmap.Control.u;
        # cot     = weight .* (sqrt(sum(((u)*(Be'*dx)).^2)).^2 / (mg * vhip));

        # state management
        state.info["mcot"] = mcot
        state.info.update(rng=rng)

        return state.replace(pipeline_state=data)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        rng, rng_noise, cmd_rng = jax.random.split(state.info["rng"], 3)

        cur_action = action
        # action = self.convert_action_to_full_dimension(action)
        # action = action * self.config.action_scale

        data0 = state.pipeline_state
        domain_idx = state.info["domain_info"]["domain_idx"]
        step_start = state.info["domain_info"]["step_start"]

        if self.config.residual_action_space:
            action = self.conv_action_based_on_idx(cur_action, jp.zeros(12))

            def update_step(step_start, domain_idx):
                new_step_start = data0.time
                new_domain_idx = 1 - domain_idx  # Switch domain_idx between 0 and 1

                return new_step_start, new_domain_idx

            def no_update(step_start, domain_idx):
                return step_start, domain_idx

            condition = (data0.time - step_start) / self.step_dur >= 1
            new_step_start, domain_idx = lax.cond(
                condition, lambda args: update_step(*args), lambda args: no_update(*args), (step_start, domain_idx)
            )

            step_start = new_step_start
            alpha = lax.cond(domain_idx == 0, lambda _: self.alpha, lambda _: jp.dot(self.R, self.alpha), None)
            q_desire = self._forward(data0.time, step_start, self.step_dur, alpha)
            action = action + q_desire

            state.info["nominal_action"] = q_desire

        if self.config.position_ctrl:
            motor_targets = jp.clip(action, self._jt_lb, self._jt_ub)
        else:
            motor_targets = jp.clip(action, self._torque_lb, self._torque_ub)

        data = self.pipeline_step(data0, motor_targets)

        # joint_angles = data.qpos[-self.model.nu :]
        joint_vel = data.qvel[-self.model.nu :]

        # observation data
        obs = self._get_obs(data, action, state.info)
        reward_tuple = self.evaluate_reward(data0, data, cur_action, state.info)
        reward = reward_tuple["total_reward"]
        reward = jp.clip(reward * self.dt, -100.0, 100.0)

        # state management
        state.info["domain_info"]["domain_idx"] = domain_idx
        state.info["domain_info"]["step_start"] = step_start
        state.info["reward_tuple"] = reward_tuple
        state.info["last_full_act"] = self.conv_action_based_on_idx(action, jp.zeros(12))
        state.info["last_act"] = cur_action
        state.info["last_vel"] = joint_vel
        state.info.update(rng=rng)

        # resetting logic if joint limits are reached or robot is falling
        # todo: verify this
        done = 0.0
        # done = jp.where(
        #     jp.logical_or(
        #         jp.any(joint_angles < 0.98 * self._jt_lb),
        #         jp.any(joint_angles > 0.98 * self._jt_ub),
        #     ),
        #     1.0,
        #     done,
        # )
        done = jp.where(data.qpos[2] < 0.7, 1.0, done)

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def evaluate_reward(self, data0: mjx.Data, data: mjx.Data, action: jp.ndarray, state_info: Dict[str, Any]) -> dict:
        """Evaluate the reward function."""
        x, xd = self._pos_vel(data)

        # Center of Mass calculations
        com_before = data.subtree_com[self.base_frame_idx]
        com_after = data.subtree_com[self.base_frame_idx]

        velocity = (com_after - com_before) / self.dt

        # Forward reward based on center of mass velocity
        forward_reward = self.config.reward.forward_reward_weight * velocity[0]

        # Healthy state reward
        min_z, max_z = self.config.healthy_z_range
        is_healthy = jp.logical_and(data.qpos[2] >= min_z, data.qpos[2] <= max_z)
        healthy_reward = (
            self.config.reward.healthy_reward
            if self.config.terminate_when_unhealthy
            else self.config.reward.healthy_reward * is_healthy
        )

        # Control cost
        ctrl_cost = self.config.reward.ctrl_cost_weight * jp.sum(jp.square(data.qfrc_actuator))
        ctrl_cost = jp.clip(ctrl_cost, -1.0, 1.0)

        # Adjusted Reward components using state.info['command']
        command = state_info["command"]
        desired_lin_vel = command[0:2]
        desired_ang_vel = command[2]

        # Tracking rewards
        tracking_lin_vel_reward = self.config.reward.tracking_lin_vel * jp.exp(
            -jp.sum(jp.square(velocity[:2] - desired_lin_vel)) / self.config.reward.tracking_sigma
        )
        tracking_ang_vel_reward = self.config.reward.tracking_ang_vel * jp.exp(
            -jp.square(data.qvel[5] - desired_ang_vel) / self.config.reward.tracking_sigma
        )  # Assuming index 5 is yaw rate

        # Regularization terms
        lin_vel_z_penalty = self.config.reward.lin_vel_z * jp.square(velocity[2])
        ang_vel_xy_penalty = self.config.reward.ang_vel_xy * jp.sum(jp.square(data.qvel[:2]))
        # orientation_penalty = self.config.reward.orientation * jp.sum(jp.square(data.xmat[2, :2]))  # Assuming this is the flatness of the base orientation
        action_rate_penalty = self.config.reward.action_rate * jp.sum(jp.square(action - state_info["last_act"]))

        # cop penalty
        cop_penalty = self.cop_reward(data, state_info)

        # Combine all reward components
        total_reward = (
            forward_reward
            + healthy_reward
            + ctrl_cost
            + tracking_lin_vel_reward
            + tracking_ang_vel_reward
            + lin_vel_z_penalty
            + ang_vel_xy_penalty
            + action_rate_penalty
            + cop_penalty
        )

        return {
            "forward_reward": forward_reward,
            "healthy_reward": healthy_reward,
            "ctrl_cost": ctrl_cost,
            "tracking_lin_vel_reward": tracking_lin_vel_reward,
            "tracking_ang_vel_reward": tracking_ang_vel_reward,
            "lin_vel_z_penalty": lin_vel_z_penalty,
            "ang_vel_xy_penalty": ang_vel_xy_penalty,
            "cop_penalty": cop_penalty,
            "action_rate_penalty": action_rate_penalty,
            "total_reward": total_reward,
        }

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
        phase_var = (data.time - step_start) / self.step_dur
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
        position = data.qpos[3:]
        velocity = data.qvel[3:]

        com2L = data.subtree_com[self.base_frame_idx] - jp.mean(data.contact.pos[0:4])
        com2R = data.subtree_com[self.base_frame_idx] - jp.mean(data.contact.pos[4:8])
        obs_list = []
        domain_idx = state_info["domain_info"]["domain_idx"]
        step_start = state_info["domain_info"]["step_start"]
        # nominal_action = state_info["nominal_action"]
        phase_var = (data.time - step_start) / self.step_dur

        # last action
        obs_list.append(position)  # 16
        obs_list.append(velocity)  # 15
        obs_list.append(self._get_contact_force(data))  # 8
        # obs_list.append(nominal_action) #12
        obs_list.append(com2L)
        obs_list.append(com2R)
        obs_list.append(state_info["last_act"])
        obs_list.append(jp.array([domain_idx, step_start, phase_var]))

        obs = jp.clip(jp.concatenate(obs_list), -100.0, 2000.0)

        # stack observations through time
        single_obs_size = len(obs)
        state_info["obs_history"] = jp.roll(state_info["obs_history"], single_obs_size)
        state_info["obs_history"] = jp.array(state_info["obs_history"]).at[:single_obs_size].set(obs)

        return state_info["obs_history"]

    def _cacc_fn(self, cacc, cdof_dot, qvel):
        if cacc is None:
            if self.sys.opt.disableflags & DisableBit.GRAVITY:
                cacc = jp.zeros((6,))
            else:
                cacc = jp.concatenate((jp.zeros((3,)), -self.sys.opt.gravity))

        cacc += jp.sum(jax.vmap(jp.multiply)(cdof_dot, qvel), axis=0)

        return cacc

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

    def _ncr(self, n, r):
        r = min(r, n - r)
        numer = jp.prod(jp.arange(n, n - r, -1))
        denom = jp.prod(jp.arange(1, r + 1))
        return numer // denom

    def _forward(self, t, t0, step_dur, alpha):
        B = 0
        tau = (t - t0) / step_dur
        tau = jp.clip(tau, 0, 1)
        for i in range(self.bez_deg + 1):
            x = self._ncr(self.bez_deg, i)
            B = B + x * ((1 - tau) ** (self.bez_deg - i)) * (tau**i) * alpha[:, i]
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

    def _getDesireJtTraj(self, alpha, step_dur, t, t0=0):
        self.q_desire = self._forward(t, t0, step_dur, alpha)
        self.dq_desire = self._forward_vel(t, t0, step_dur, alpha)

    def _remap_coeff(self):
        # assuming the last num_output is the only relevant ones
        baseRemap = jp.array([1, -1, 1, -1, 1, -1], dtype=jp.float32)
        legRemap = jp.array([-1, -1, 1, 1, 1, -1], dtype=jp.float32)
        relabel = jax.scipy.linalg.block_diag(jp.diag(baseRemap), jp.diag(legRemap), jp.diag(legRemap))
        relabelIdx = jp.array([0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11], dtype=jp.int32)
        R = jp.zeros_like(relabel)
        R = R.at[relabelIdx].set(relabel)
        return R
