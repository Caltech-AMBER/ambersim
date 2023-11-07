import math
import os
from dataclasses import dataclass

import jax
import jax.numpy as jp
import mujoco
from brax.envs.base import PipelineEnv, State
from mujoco.mjx._src import scan

from ambersim import ROOT
from ambersim.base import MjxEnv


@dataclass
class ExoConfig:
    """config dataclass that specified the reward coefficient and other custom setting for exo env."""

    forward_reward_weight: float = 1.25
    ctrl_cost_weight: float = 0.1
    healthy_reward: float = 5.0
    terminate_when_unhealthy: bool = True
    healthy_z_range: tuple = (0.85, 1)
    reset_noise_scale: float = 1e-2
    exclude_current_positions_from_observation: bool = True
    backend: str = "generalized"
    xml_file: str = "loadedExo.xml"
    rand_terrain: bool = True
    physics_steps_per_control_step: int = 5


class Exo(MjxEnv):
    """custom environment for the exoskeleton."""

    def __init__(self, exo_config: ExoConfig = ExoConfig, **kwargs):
        """Initialize the environment."""
        if exo_config.rand_terrain:
            xml_file = "loadedExo_no_terrain.xml"
            path = os.path.join(ROOT, "models", "exo", xml_file)
            # path = self.genRandBoxes(path,grid_size = (2,5))
        else:
            path = os.path.join(ROOT, "models", "exo", exo_config.xml_file)

        # sys = mjcf.load(path)

        mj_model = mujoco.MjModel.from_xml_path(path)

        physics_steps_per_control_step = 5
        kwargs["physics_steps_per_control_step"] = kwargs.get(
            "physics_steps_per_control_step", physics_steps_per_control_step
        )

        super().__init__(mj_model=mj_model, **kwargs)
        # super().__init__()  # Initialize the base class without any arguments.

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, jp.zeros(self.sys.nu))
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
            "ang_m0": zero,
            "ang_m1": zero,
            "ang_m2": zero,
        }
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        com_before = data0.subtree_com[2]
        com_after = data.subtree_com[2]

        # mujoco.mjx.com_vel(self.sys,data)
        # com_vel = data.cvel[2][0:3]
        com_acc = scan.body_tree(self.sys, self.cacc_fn, "vv", "b", data.cdof_dot, data.qvel)

        x, xd = self._pos_vel(data)
        base_ang_vel = math.rotate(xd.ang[1], math.quat_inv(x.rot[1]))
        ang_m = base_ang_vel * self.sys.body_inertia[2]

        velocity = (com_after - com_before) / self.dt
        forward_reward = self._forward_reward_weight * velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.qpos[2] < min_z, x=0.0, y=1.0)
        is_healthy = jp.where(data.qpos[2] > max_z, x=0.0, y=is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(data, action)
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
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
            x_acceleration=com_acc[0],
            y_acceleration=com_acc[1],
            z_acceleration=com_acc[2],
            ang_m0=ang_m[0],
            ang_m1=ang_m[1],
            ang_m2=ang_m[2],
        )

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)
