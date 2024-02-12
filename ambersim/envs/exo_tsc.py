import os

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
from brax.math import quat_from_3x3
from jax import lax
from jax.scipy.integrate import trapezoid
from jax.scipy.linalg import solve
from mujoco import _enums, _structs, mjx
from mujoco.mjx._src import scan
from mujoco.mjx._src.forward import _integrate_pos, fwd_position
from mujoco.mjx._src.math import norm, quat_sub, rotate
from mujoco.mjx._src.types import DisableBit

from ambersim import ROOT
from ambersim.base import MjxEnv
from ambersim.Controller.KinematicsInterface import CoMKinematics, GeomKinematics, SiteKinematics
from ambersim.envs.exo_base import BehavState, Exo, ExoConfig, StanceState
from ambersim.utils.asset_utils import add_geom_to_env, add_heightfield_to_mujoco_xml, generate_boxes_xml
from ambersim.utils.io_utils import set_actuators_type


class Exo_TSC(Exo):
    """custom environment for the exoskeleton."""

    def __init__(self, config: ExoConfig):
        """Initialize the environment."""
        config.xml_file = "Maegan_Tucker_loadedExo.xml"
        super().__init__(config)

        self._config = config
        self.site_ids = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, site_name)
            for site_name in ["left_sole_frame", "right_sole_frame"]
        ]
        self.site_kinematics = [SiteKinematics(self.model, site_id) for site_id in self.site_ids]
        # self.geom_ids = self.foot_geom_idx
        # self.geom_kinematics = [GeomKinematics(self.model, geom_id) for geom_id in self.foot_geom_idx]
        self.com_kinematics = CoMKinematics(self.model, self.base_frame_idx)
        self.ik_max_iter = 20

        self.load_output_traj()

    def load_output_traj(self):
        """Load the task space trajectory."""
        gait_params_file = os.path.join(ROOT, "..", "models", "exo", "jt_bez_2023-09-10.yaml")
        with open(gait_params_file, "r") as file:
            gait_params = yaml.safe_load(file)

        coeff_output = np.reshape(np.array(gait_params["coeff_output"]), (12, 8), order="F")
        coeff_output[6, :] = [0, 0, 0, 0, 1, 1, 1, 1]
        coeff_output[7, :] = [0, 0, 0, 0, 1, 1, 1, 1]

        coeff_output[9:12, :] = 0

        self.alpha = jp.array(coeff_output)

        # override alpha to be 0 for orientation

        self._q_init = jp.concatenate([jp.array(gait_params["ffPos"]), jp.array(gait_params["startingPos"])], axis=0)
        self._dq_init = jp.concatenate([jp.array(gait_params["ffVel"]), jp.array(gait_params["startingVel"])], axis=0)

        # not mapping sw ft y output
        output_sign = jp.array([1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1])
        self.R = jp.diag(output_sign)

        self.des_SL = jp.array([gait_params["SL"], gait_params["SW"]])

        return

    def getWalkingNomDes(self, state: State) -> jp.ndarray:
        """Get the nominal desire for walking."""
        data0 = state.pipeline_state
        domain_idx = state.info["domain_info"]["domain_idx"]
        step_start = state.info["domain_info"]["step_start"]

        def update_step(step_start, domain_idx):
            new_step_start = data0.time
            new_domain_idx = 1 - domain_idx  # Switch domain_idx between 0 and 1
            return new_step_start, new_domain_idx

        def no_update(step_start, domain_idx):
            return step_start, domain_idx

        condition = (data0.time - step_start) / self.step_dur[BehavState.Walking] >= 1
        if self.config.impact_based_switching:

            condition = jp.logical_and(
                self.checkImpact(data0, state.info),
                (data0.time - step_start) / self.step_dur[BehavState.Walking] >= 0.8,
            )

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

        y_desire = self._forward(data0.time, step_start, self.step_dur[BehavState.Walking], alpha)

        state.info["domain_info"]["domain_idx"] = domain_idx
        state.info["domain_info"]["step_start"] = step_start

        y_desire = y_desire.at[2].set(y_desire[2] - 0.005)
        jax.debug.print("shift z downwards")

        ya = self.getY(data0, domain_idx)
        new_pswing0 = lax.cond(condition, lambda _: ya[6:8], lambda _: state.info["sw_ft_pos0"], None)
        state.info["sw_ft_pos0"] = new_pswing0

        # override swing foot position
        p_des = lax.cond(
            domain_idx == StanceState.Right.value,
            lambda _: state.info["des_step_pos"],
            lambda _: jp.array([state.info["des_step_pos"][0], -1 * state.info["des_step_pos"][1]]),
            None,
        )
        jax.debug.print("p_des: {}", p_des)
        y_desire = y_desire.at[6].set((1 - y_desire[6]) * new_pswing0[0] + y_desire[6] * p_des[0])
        jax.debug.print("p_des: {}", p_des)
        y_desire = y_desire.at[7].set((1 - y_desire[7]) * new_pswing0[1] + y_desire[7] * p_des[1])

        return y_desire, state, condition

    def getY(self, data, whichStance):
        """Get task space output for the exoskeleton."""
        # get CoM, Stance, Swing Ft position and orientation
        left_foot_pos = self.site_kinematics[0].getSitePos(data)
        left_foot_ori = self.site_kinematics[0].getSiteOri(data)
        right_foot_pos = self.site_kinematics[1].getSitePos(data)
        right_foot_ori = self.site_kinematics[1].getSiteOri(data)

        # based on whichStance assign the stance and swing ft using lax.cond
        swing_ft_pos, swing_ft_ori, stance_ft_pos, stance_ft_ori = lax.cond(
            whichStance == StanceState.Right.value,
            lambda _: (left_foot_pos, left_foot_ori, right_foot_pos, right_foot_ori),
            lambda _: (right_foot_pos, right_foot_ori, left_foot_pos, left_foot_ori),
            None,
        )

        y = jp.zeros(18)
        y = y.at[0:3].set(self.com_kinematics.getPos(data) - stance_ft_pos)
        y = y.at[3:6].set(self.quat2eulXYZ(data.qpos[3:7]))
        # y_pos = y_pos.at[0:3].set(data.subtree_com[env.base_frame_idx]) - stance_ft_pos
        y = y.at[6:9].set(swing_ft_pos - stance_ft_pos)
        y = y.at[9:12].set(self.quat2eulXYZ(swing_ft_ori))
        y = y.at[12:15].set(stance_ft_pos)
        y = y.at[15:18].set(self.quat2eulXYZ(stance_ft_ori))

        return y

    def getJacobian(self, data, whichStance):
        """Get task space jacobian for the exoskeleton."""
        J_l_p, J_l_r = self.site_kinematics[0].JacSite(data)
        J_r_p, J_r_r = self.site_kinematics[1].JacSite(data)

        # based on which stance, assign the stance and swing ft jacobians using lax.cond
        jac_swp, jac_swr, hol_jacp, hol_jacr = lax.cond(
            whichStance == StanceState.Right.value,
            lambda _: (J_l_p, J_l_r, J_r_p, J_r_r),
            lambda _: (J_l_p, J_l_r, J_r_p, J_r_r),
            None,
        )

        jacp_com = self.com_kinematics.JacSubtreeCoM(data) - hol_jacp
        jac_swp = jac_swp - hol_jacp

        jacr_pelvis = self.com_kinematics.JacPelvOri(data)
        # concatenate the jacobians (order: com2st, pelv ori, sw2st, sw ori)
        jac = jp.concatenate([jacp_com, jacr_pelvis, jac_swp, jac_swr, hol_jacp, hol_jacr], axis=0)

        return jac

    def solveIK(self, state: State, y_des: jp.ndarray) -> jp.ndarray:
        """Solves the IK problem."""
        whichStance = state.info["domain_info"]["domain_idx"]
        q = state.info["joint_desire"]
        data = state.pipeline_state
        damping_factor = 1e-3

        y_des_full = jp.zeros(18)
        y_des_full = y_des_full.at[0:12].set(y_des[0:12])

        def body_fun(step, loop_vars):
            q, data, err, success = loop_vars
            # Assuming fwd_position is a function that updates and returns data based on current q
            data = data.replace(qpos=q)
            data = fwd_position(self.sys, data)  # Make sure this is JAX-compatible
            y_cur = self.getY(data, whichStance)

            err = y_des_full - y_cur

            # jax.debug.print('qpos" {}',data.qpos)
            # jax.debug.print('step: {}',step)
            # jax.debug.print('err: {}',err)
            # jax.debug.print('y_cur: {}',y_cur)
            # jax.debug.print('y_des_full: {}',y_des_full)

            new_success = jp.max(jp.abs(err)) < 1e-3

            def update_q(_):
                J = self.getJacobian(data, whichStance)
                JJt = J @ J.T + jp.eye(J.shape[0]) * damping_factor
                v = J.T @ solve(JJt, err, assume_a="pos")

                q_new = _integrate_pos(self.model.jnt_type, data.qpos, v, 1)  # Ensure this matches your implementation

                return q_new

            q_new = lax.cond(new_success, lambda _: q, update_q, None)
            # Update data.qpos with q_new here as needed

            # Update data for the next iteration
            data = data.replace(qpos=q_new)

            return q_new, data, err, new_success | success

        # Initial loop variables: initial q, initial data, initial success status
        data = data.replace(qpos=q, qvel=jp.zeros(self.model.nv), ctrl=jp.zeros(self.model.nu))
        init_loop_vars = (q, data, jp.ones(18), False)

        q_final, data_final, err, success = jax.lax.fori_loop(0, self.ik_max_iter, body_fun, init_loop_vars)

        return q_final, err, success

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        # cur_action = action

        data0 = state.pipeline_state
        ya = self.getY(data0, state.info["domain_info"]["domain_idx"])

        jax.debug.print("ya: {}", ya)

        y_desire, state, condition = self.getWalkingNomDes(state)
        jax.debug.print("y_desire: {}", y_desire)

        scaled_action = self.config.action_scale * action
        if self.config.residual_action_space:
            action = y_desire.at[self.config.custom_act_idx].set(y_desire[self.config.custom_act_idx] + scaled_action)
        else:
            action = y_desire + scaled_action

        jt_action, err, sucess = self.solveIK(state, action)
        jax.debug.print("err: {}", err)
        jax.debug.print("sucess: {}", sucess)
        q_desire = jt_action
        q_actual = data0.qpos[-self.model.nu :]
        new_offset = lax.cond(
            condition, lambda _: q_actual - q_desire[-self.model.nu :], lambda _: state.info["offset"], None
        )
        state.info["offset"] = new_offset

        blended_action = self.jt_blending(
            data0.time, self.step_dur[state.info["state"]], jt_action[-self.model.nu :], state.info
        )
        motor_targets = jp.clip(blended_action, self._jt_lb, self._jt_ub)

        if self.config.traj_opt:
            # if data0.time > 0.5 and data0.done, do not step; just return as is
            # use lax.cond to avoid jax error
            def true_fun(_):
                return data0

            def false_fun(_):
                return self.pipeline_step(data0, motor_targets)

            cond = jp.logical_and(data0.time > 0.5, state.done)
            data = lax.cond(cond, true_fun, false_fun, None)

        else:
            data = self.pipeline_step(data0, motor_targets)

        # observation data
        # reward_tuple = self.evaluate_reward(data0, data, cur_action, state.info)

        # state management
        # state.info["reward_tuple"] = reward_tuple

        y_des_full = jp.zeros(18)
        y_des_full = y_des_full.at[0:12].set(action[0:12])

        state.info["blended_action"] = blended_action
        state.info["joint_desire"] = jt_action
        state.info["ya"] = ya
        state.info["yd"] = y_des_full
        state.info["output_err"] = y_des_full - ya
        state.info["tracking_err"] = q_desire[-self.model.nu :] - q_actual
        done = self.checkDone(data)

        # reward = jp.sum(jp.array(list(reward_tuple.values())))
        # reward = reward + (1 - done) * self.config.reward.healthy_reward

        return state.replace(pipeline_state=data, done=done)

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

        qpos, qvel = lax.cond(behavstate == BehavState.Walking, true_fun, false_fun, None)

        if not self.config.no_noise:
            qpos = qpos + qpos_noise
            qvel = qvel + qvel_noise

        if self.config.slope:
            qpos = qpos.at[2].set(qpos[2] + 0.005)

        if self.config.rand_terrain:
            # check the position of z for geom 0,1
            # and set  qpos[2] accordingly
            zpos_offset = (
                self.sys.geom_size[self.terrain_geom_idx[0], 2]
                + self.sys.geom_pos[self.terrain_geom_idx[0], 2]
                - (-0.005)
            )
            qpos = qpos.at[2].set(qpos[2] + zpos_offset)
        data = self.pipeline_init(qpos, qvel)

        p_swing0 = self.getY(data, StanceState.Right.value)[6:8]
        reward, done, zero = jp.zeros(3)

        state_info = {
            "state": behavstate,
            "offset": jp.zeros(12),
            "sw_ft_pos0": p_swing0,
            "domain_info": {
                "step_start": zero,
                "domain_idx": StanceState.Right.value,
            },
            "blended_action": jp.zeros(12),
            "joint_desire": self._q_init,
            "ya": jp.zeros(18),
            "yd": jp.zeros(18),
            "last_action": jp.zeros(self.custom_act_space_size),
            "tracking_err": jp.zeros(18),
            "output_err": jp.zeros(18),
            "alpha": self.alpha,
            "des_step_pos": self.des_SL,
            "impact_threshold": self.config.impact_threshold,
        }

        obs = []

        metrics = {}

        return State(data, obs, reward, done, metrics, state_info)


if __name__ == "__main__":
    config = ExoConfig()
    config.impact_based_switching = False
    config.physics_steps_per_control_step = 1
    env = Exo_TSC(config)
    state = env.reset(jax.random.PRNGKey(0), behavstate=BehavState.Walking)
    action = jp.zeros(12)

    env.step(state, jp.zeros(env.action_size))

    jit_env_step = jax.jit(env.step)

    output_video = "tsc_nominal_policy.mp4"
    num_steps = 100
    """Run the simulation basic version."""
    env.getRender()

    images = []
    logged_data_per_step = []
    for _ in range(num_steps):
        state = jit_env_step(state, jp.zeros(env.action_size))
        images.append(env.get_image(state.pipeline_state))
        logged_data = {}
        logged_data = env.log_state_info(
            state.info, ["domain_info", "tracking_err", "joint_desire", "ya", "yd"], logged_data
        )

        logged_data_per_step.append(logged_data)

    media.write_video(output_video, images, fps=1.0 / env.dt)

    env.plot_logged_data(logged_data_per_step, save_dir="/home/kli5/ambersim/ambersim/envs/plots")

    # todo check conversion for frost euler angle for pelvis orientation
    # check yd full order matching
