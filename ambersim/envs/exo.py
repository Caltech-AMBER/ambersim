import os
from dataclasses import dataclass

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import yaml
from brax.envs.base import PipelineEnv, State
from jax import lax
from mujoco import _enums, _structs, mjx
from mujoco.mjx._src import scan
from mujoco.mjx._src.types import DisableBit

from ambersim import ROOT
from ambersim.base import MjxEnv
from ambersim.utils.io_utils import set_actuators_type


@dataclass
class ExoConfig:
    """config dataclass that specified the reward coefficient and other custom setting for exo env."""

    forward_reward_weight: float = 1.25
    ctrl_cost_weight: float = 0.1
    healthy_reward: float = 5.0
    terminate_when_unhealthy: bool = True
    reset_2_stand: bool = False
    healthy_z_range: tuple = (0.85, 1)
    reset_noise_scale: float = 1e-2
    exclude_current_positions_from_observation: bool = True
    xml_file: str = "loadedExo.xml"
    jt_traj_file: str = "jt_bez_2023-09-10.yaml"
    loading_pos_file: str = "sim_config_loadingPos.yaml"
    ctrl_limit_file: str = "limits.yaml"
    rand_terrain: bool = False
    position_ctrl: bool = True
    residual_action_space: bool = True
    physics_steps_per_control_step: int = 10


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
            path = os.path.join(ROOT, "models", "exo", config.xml_file)

        self.model = mujoco.MjModel.from_xml_path(path)
        self.load_traj()
        self.load_ctrl_params()

        # TODO: check where the base frame is

        super().__init__(mj_model=self.model, physics_steps_per_control_step=self.config.physics_steps_per_control_step)

    def load_traj(self) -> None:
        """Load default trajectory from yaml file specfied in the config."""
        gait_params_file = os.path.join(ROOT, "models", "exo", self.config.jt_traj_file)
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
        loading_pos_file = os.path.join(ROOT, "models", "exo", self.config.loading_pos_file)
        with open(loading_pos_file, "r") as file:
            load_params = yaml.safe_load(file)

        self._q_load = jp.concatenate([jp.array(load_params["ffPos"]), jp.array(load_params["startingPos"])], axis=0)

    def load_ctrl_params(self) -> None:
        """Load joint limit and config limit from yaml file specfied in the config."""
        limit_file = os.path.join(ROOT, "models", "exo", self.config.ctrl_limit_file)
        with open(limit_file, "r") as file:
            ctrl_limit = yaml.safe_load(file)
        self._p_gains = jp.array(ctrl_limit["joint"]["kp"])
        self._d_gains = jp.array(ctrl_limit["joint"]["kd"])

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

        obs = self._get_obs(data, jp.zeros(self.sys.nu), 0.0, 0.0)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "forward_reward": zero,
            "reward_linvel": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "x_acceleration": zero,
            "y_acceleration": zero,
            "z_acceleration": zero,
            # "ang_m0": zero,
            # "ang_m1": zero,
            # "ang_m2": zero,
        }
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        step_start = state.obs[0]
        domain_idx = state.obs[1]

        if self.config.residual_action_space:

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

        data = self.pipeline_step(data0, action)

        com_before = data0.subtree_com[1]
        com_after = data.subtree_com[1]

        # mujoco.mjx.com_vel(self.sys,data)
        # com_vel = data.cvel[2][0:3]
        com_acc = scan.body_tree(self.sys, self._cacc_fn, "vv", "b", data.cdof_dot, data.qvel)

        # x, xd = self._pos_vel(data)
        # base_ang_vel = math.rotate(xd.ang[1], math.quat_inv(x.rot[1]))
        # ang_m = base_ang_vel * self.sys.body_inertia[2]

        velocity = (com_after - com_before) / self.dt
        forward_reward = self.config.forward_reward_weight * velocity[0]

        min_z, max_z = self.config.healthy_z_range
        is_healthy = jp.where(data.qpos[2] < min_z, x=0.0, y=1.0)
        is_healthy = jp.where(data.qpos[2] > max_z, x=0.0, y=is_healthy)
        if self.config.terminate_when_unhealthy:
            healthy_reward = self.config.healthy_reward
        else:
            healthy_reward = self.config.healthy_reward * is_healthy

        #
        ctrl_cost = self.config.ctrl_cost_weight * jp.sum(jp.square(data.qfrc_actuator))

        obs = self._get_obs(data, action, step_start, domain_idx)
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self.config.terminate_when_unhealthy else 0.0
        state.metrics.update(
            forward_reward=forward_reward,
            reward_linvel=forward_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            x_position=com_after[0],
            y_position=com_after[1],
            distance_from_origin=jp.linalg.norm(com_after),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            x_acceleration=com_acc[1, 0],
            y_acceleration=com_acc[1, 1],
            z_acceleration=com_acc[1, 2],
            # ang_m0=ang_m[0],
            # ang_m1=ang_m[1],
            # ang_m2=ang_m[2],
        )

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data, action: jp.ndarray, step_start: float, domain_idx: float) -> jp.ndarray:
        """Observes position, velocities."""
        position = data.qpos
        if self.config.exclude_current_positions_from_observation:
            position = position[2:]

        # external_contact_forces are excluded
        return jp.concatenate(
            [
                jp.array([step_start]),
                jp.array([domain_idx]),
                position,
                data.qvel,
                data.cinert[1:].ravel(),
                data.cvel[1:].ravel(),
                data.qfrc_actuator,
            ]
        )

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
        camera = mujoco.MjvCamera()
        camera.azimuth = 0
        camera.elevation = 0
        camera.distance = 3
        camera.lookat = jp.array([0, 0, 0.5])
        self.camera = camera

        renderer = mujoco.Renderer(self.model, 480, 640)
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
