from ambersim.base import MjxEnv, State
from typing import Tuple, Dict, Any

import jax
from jax import numpy as jp
from brax import math
from pathlib import Path
from ambersim import ROOT
from ambersim.utils.asset_utils import load_mjx_model_from_file
import mujoco

from brax.base import Base, Motion, Transform
from flax import struct


@struct.dataclass
class A1RewardConfig:
    """
    Weightings for the reward function.
    """

    # Tracking rewards are computed using exp(-delta^2/sigma)
    # sigma can be a hyperparameters to tune.
    # Track the base x-y velocity (no z-velocity tracking.)
    tracking_lin_vel: float = 1.5

    # Track the angular velocity along z-axis, i.e. yaw rate.
    tracking_ang_vel: float = 0.8

    # Below are regularization terms, we roughly divide the
    # terms to base state regularizations, joint
    # regularizations, and other behavior regularizations.
    # Penalize the base velocity in z direction, L2 penalty.
    lin_vel_z: float = -2.0

    # Penalize the base roll and pitch rate. L2 penalty.
    ang_vel_xy: float = -0.05

    # Penalize non-zero roll and pitch angles. L2 penalty.
    orientation: float = -5.0

    # L2 regularization of joint torques, |tau|^2.
    # torques=-0.0002,
    torques: float = -0.002

    # Penalize the change in the action and encourage smooth
    # actions. L2 regularization |action - last_action|^2
    action_rate: float = -0.1

    # Encourage long swing steps.  However, it does not
    # encourage high clearances.
    feet_air_time: float = 0.2

    # Encourage no motion at zero command, L2 regularization
    # |q - q_default|^2.
    stand_still: float = -0.5

    # Early termination penalty.
    termination: float = -1.0

    # Penalizing foot slipping on the ground.
    foot_slip: float = -0.1

    # Tracking reward = exp(-error^2/sigma).
    tracking_sigma: float = 0.25


@struct.dataclass
class A1CommandConfig:
    """Hyperparameters for random commands for A1."""

    lin_vel_x: tuple[float] = (-0.6, 1.0)  # min max [m/s]
    lin_vel_y: tuple[float] = (-0.8, 0.8)  # min max [m/s]
    ang_vel_yaw: tuple[float] = (-0.7, 0.7)  # min max [rad/s]


@struct.dataclass
class A1Config:
    reward: A1RewardConfig = A1RewardConfig()
    command: A1CommandConfig = A1CommandConfig()
    # Scale for uniform noise added to observations.
    obs_noise: float = 0.05
    # Scaling for action space.
    action_scale: float = 0.3
    # Max episode length.
    reset_horizon: int = 500
    # Lower joint limits.
    joint_lowers: jp.ndarray = struct.field(default_factory=lambda: jp.array([-0.802851, -1.0472, -2.69653] * 4))
    # Upper joint limits.
    joint_uppers: jp.ndarray = struct.field(default_factory=lambda: jp.array([0.802851, 4.18879, -0.916298] * 4))
    # Default joint positions for standing.
    standing_config: jp.ndarray = struct.field(
        default_factory=lambda: jp.array([0, 0, 0.27, 1, 0, 0, 0] + [0, 0.9, -1.8] * 4)
    )
    # Model path
    model_path: Path = Path(ROOT) / "models" / "cursed_a1" / "scene.xml"
    # Number of env steps per command.
    physics_steps_per_control_step: int = 10
    # Body index of torso.
    torso_index: int = 1
    # Body indices of the feet.
    feet_indices: jp.ndarray = struct.field(default_factory=lambda: jp.array([3, 6, 9, 12]))
    # Positions of feet relative to last links.
    feet_pos: jp.ndarray = struct.field(
        default_factory=lambda: jp.array(
            [
                [0.0, 0.0, -0.2],
                [0.0, 0.0, -0.2],
                [0.0, 0.0, -0.2],
                [0.0, 0.0, -0.2],
            ]
        )
    )


class A1Env(MjxEnv):
    def __init__(self, config: A1Config = A1Config()):
        self.config = config

        # Load model.
        mj_model = load_mjx_model_from_file(config.model_path)

        # Force CG solver for compatibility with MJX.
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 3
        mj_model.opt.ls_iterations = 4

        super().__init__(
            mj_model,
            config.physics_steps_per_control_step,
        )

        self._init_q = self.model.keyframe("home").qpos
        self._default_ap_pose = self.model.keyframe("home").qpos[7:]

    def sample_command(self, rng: jax.Array):
        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=self.config.command.lin_vel_x[0], maxval=self.config.command.lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=self.config.command.lin_vel_y[0], maxval=self.config.command.lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=self.config.command.ang_vel_yaw[0], maxval=self.config.command.ang_vel_yaw[1]
        )
        new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_cmd

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)

        qpos = jp.array(self._init_q)
        qvel = jp.zeros(self.model.nv)
        new_cmd = self.sample_command(key)
        data = self.pipeline_init(qpos, qvel)

        state_info = {
            "rng": rng,
            "last_act": jp.zeros(12),
            "last_vel": jp.zeros(12),
            "last_contact_buffer": jp.zeros((20, 4), dtype=bool),
            "command": new_cmd,
            "last_contact": jp.zeros(4, dtype=bool),
            "feet_air_time": jp.zeros(4),
            "obs_history": jp.zeros(15 * 31),
            "reward_tuple": {
                "tracking_lin_vel": 0.0,
                "tracking_ang_vel": 0.0,
                "lin_vel_z": 0.0,
                "ang_vel_xy": 0.0,
                "orientation": 0.0,
                "torque": 0.0,
                "action_rate": 0.0,
                "stand_still": 0.0,
                "feet_air_time": 0.0,
                "foot_slip": 0.0,
            },
            "step": 0,
        }

        x, xd = self._pos_vel(data)
        obs = self._get_obs(data.qpos, x, xd, state_info)
        reward, done = jp.zeros(2)

        metrics = {"total_dist": 0.0}
        for k in state_info["reward_tuple"]:
            metrics[k] = state_info["reward_tuple"][k]

        state = State(data, obs, reward, done, metrics, state_info)

        return state

    def step(self, state: State, action: jax.Array) -> State:
        rng, rng_noise, cmd_rng = jax.random.split(state.info["rng"], 3)

        # physics step
        cur_action = jp.array(action)
        action = action[:12] * self.config.action_scale
        motor_targets = jp.clip(action + self._default_ap_pose, self.config.joint_lowers, self.config.joint_uppers)
        data = self.pipeline_step(state.pipeline_state, motor_targets)

        # observation data
        x, xd = self._pos_vel(data)
        obs = self._get_obs(data.qpos, x, xd, state.info)
        obs_noise = self.config.obs_noise * jax.random.uniform(rng_noise, obs.shape, minval=-1, maxval=1)
        qpos, qvel = data.qpos, data.qvel
        joint_angles = qpos[7:]
        joint_vel = qvel[6:]

        # foot contact data based on z-position
        foot_contact = 0.02 - self._get_feet_pos_vel(x, xd)[0][:, 2]
        contact = foot_contact > -1e-3  # a mm or less off the floor
        contact_filt_mm = jp.logical_or(contact, state.info["last_contact"])
        contact_filt_cm = jp.logical_or(foot_contact > -1e-2, state.info["last_contact"])
        first_contact = (state.info["feet_air_time"] > 0) * (contact_filt_mm)
        state.info["feet_air_time"] += self.dt

        # reward
        reward_tuple = {
            "tracking_lin_vel": (
                self._reward_tracking_lin_vel(state.info["command"], x, xd) * self.config.reward.tracking_lin_vel
            ),
            "tracking_ang_vel": (
                self._reward_tracking_ang_vel(state.info["command"], x, xd) * self.config.reward.tracking_ang_vel
            ),
            "lin_vel_z": (self._reward_lin_vel_z(xd) * self.config.reward.lin_vel_z),
            "ang_vel_xy": (self._reward_ang_vel_xy(xd) * self.config.reward.ang_vel_xy),
            "orientation": (self._reward_orientation(x) * self.config.reward.orientation),
            "torque": (self._reward_torques(data.qfrc_actuator) * self.config.reward.torques),
            "action_rate": (
                self._reward_action_rate(cur_action, state.info["last_act"]) * self.config.reward.action_rate
            ),
            "stand_still": (
                self._reward_stand_still(state.info["command"], joint_angles, self._default_ap_pose)
                * self.config.reward.stand_still
            ),
            "feet_air_time": (
                self._reward_feet_air_time(
                    state.info["feet_air_time"],
                    first_contact,
                    state.info["command"],
                )
                * self.config.reward.feet_air_time
            ),
            "foot_slip": (self._reward_foot_slip(x, xd, contact_filt_cm) * self.config.reward.foot_slip),
        }
        reward = sum(reward_tuple.values())
        reward = jp.clip(reward * self.dt, 0.0, 10000.0)

        # state management
        state.info["last_act"] = cur_action
        state.info["last_vel"] = joint_vel
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact
        state.info["last_contact_buffer"] = jp.roll(state.info["last_contact_buffer"], 1, axis=0)
        state.info["last_contact_buffer"] = state.info["last_contact_buffer"].at[0].set(contact)
        state.info["reward_tuple"] = reward_tuple
        state.info["step"] += 1
        state.info.update(rng=rng)

        # resetting logic if joint limits are reached or robot is falling
        done = 0.0
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.where(jp.dot(math.rotate(up, x.rot[0]), up) < 0, 1.0, done)
        done = jp.where(
            jp.logical_or(
                jp.any(joint_angles < 0.98 * self.config.joint_lowers),
                jp.any(joint_angles > 0.98 * self.config.joint_uppers),
            ),
            1.0,
            done,
        )
        done = jp.where(x.pos[self.config.torso_index, 2] < 0.18, 1.0, done)

        # termination reward
        reward += jp.where(
            (done == 1.0) & (state.info["step"] < self.config.reset_horizon),
            self.config.reward.termination,
            0.0,
        )

        # when done, sample new command if more than reset_horizon timesteps
        # achieved
        state.info["command"] = jp.where(
            (done == 1.0) & (state.info["step"] > self.config.reset_horizon),
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        # reset the step counter when done
        state.info["step"] = jp.where(
            (done == 1.0) | (state.info["step"] > self.config.reset_horizon), 0, state.info["step"]
        )

        # log total displacement as a proxy metric
        state.metrics["total_dist"] = math.normalize(x.pos[self.config.torso_index])[1]
        for k in state.info["reward_tuple"].keys():
            state.metrics[k] = state.info["reward_tuple"][k]

        state = state.replace(pipeline_state=data, obs=obs + obs_noise, reward=reward, done=done)
        return state

    def _get_obs(self, qpos: jax.Array, x: Transform, xd: Motion, state_info: Dict[str, Any]) -> jax.Array:
        # Get observations:
        # yaw_rate,  projected_gravity, command,  motor_angles, last_action

        inv_base_orientation = math.quat_inv(x.rot[0])
        local_rpyrate = math.rotate(xd.ang[0], inv_base_orientation)
        cmd = state_info["command"]

        obs_list = []
        # yaw rate
        obs_list.append(jp.array([local_rpyrate[2]]) * 0.25)
        # projected gravity
        obs_list.append(math.rotate(jp.array([0.0, 0.0, -1.0]), inv_base_orientation))
        # command
        obs_list.append(cmd * jp.array([2.0, 2.0, 0.25]))
        # motor angles
        angles = qpos[7:19]
        obs_list.append(angles - self._default_ap_pose)
        # last action
        obs_list.append(state_info["last_act"])

        obs = jp.clip(jp.concatenate(obs_list), -100.0, 100.0)

        # stack observations through time
        single_obs_size = len(obs)
        state_info["obs_history"] = jp.roll(state_info["obs_history"], single_obs_size)
        state_info["obs_history"] = jp.array(state_info["obs_history"]).at[:single_obs_size].set(obs)
        return state_info["obs_history"]

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jp.sum(jp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _reward_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        # Penalize changes in actions
        return jp.sum(jp.square(act - last_act))

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

    def _reward_feet_air_time(self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array) -> jax.Array:
        # Reward air time.
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= math.normalize(commands[:2])[1] > 0.05  # no reward for zero command
        return rew_air_time

    def _reward_stand_still(self, commands: jax.Array, joint_angles: jax.Array, default_angles: jax.Array) -> jax.Array:
        # Penalize motion at zero commands
        return jp.sum(jp.abs(joint_angles - default_angles)) * (math.normalize(commands[:2])[1] < 0.1)

    def _get_feet_pos_vel(self, x: Transform, xd: Motion) -> Tuple[jax.Array, jax.Array]:
        offset = Transform.create(pos=self.config.feet_pos)
        pos = x.take(self.config.feet_indices).vmap().do(offset).pos
        vel = offset.vmap().do(xd.take(self.config.feet_indices)).vel
        return pos, vel

    def _reward_foot_slip(self, x: Transform, xd: Motion, contact_filt: jax.Array) -> jax.Array:
        # Get feet velocities
        _, foot_world_vel = self._get_feet_pos_vel(x, xd)
        # Penalize large feet velocity for feet that are in contact with the ground.
        return jp.sum(jp.square(foot_world_vel[:, :2]) * contact_filt.reshape((-1, 1)))
