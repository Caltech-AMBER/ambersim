from ambersim.base import MjxEnv
from dataclasses import dataclass
from typing import Tuple

import jax
from jax import numpy as jp
from pathlib import Path
from ambersim import ROOT
from ambersim.utils.asset_utils import load_mjx_model_from_file


@dataclass
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
    foot_slip: float = (-0.1,)
    # Tracking reward = exp(-error^2/sigma).
    tracking_sigma: float = 0.25


@dataclass
class A1CommandConfig:
    """Hyperparameters for random commands for A1."""

    #
    lin_vel_x: tuple[float] = (-0.6, 1.0)  # min max [m/s]
    lin_vel_y: tuple[float] = (-0.8, 0.8)  # min max [m/s]
    ang_vel_yaw: tuple[float] = (-0.7, 0.7)  # min max [rad/s]


@dataclass
class A1Config:mj_model.
    reward_config: A1RewardConfig = A1RewardConfig()
    command_config: A1CommandConfig = A1CommandConfig()
    # Scale for uniform noise added to observations.
    obs_noise: float = 0.05
    # Scaling for action space.
    action_scale: float = 0.3
    # Max episode length.
    reset_horizon: int = 500
    # Lower joint limits.
    joint_lowers: jp.ndarray = jp.array([-0.802851, -1.0472, -2.69653] * 4)
    # Upper joint limits.
    joint_uppers: jp.ndarray = jp.array([0.802851, 4.18879, -0.916298] * 4)
    # Default joint positions for standing.
    standing_config: jp.ndarray = jp.array([0, 0, 0.27, 1, 0, 0, 0] + [0, 0.9, -1.8] * 4)
    # Model path
    model_path: Path = Path(ROOT) / "models" / "cursed_a1" / "a1.xml"
    # Number of env steps per command.
    physics_steps_per_control_step: int = 10
    # Body index of torso.
    torso_index: int = 1
    # Body indices of the feet.
    foot_indices: Tuple[int] = (4, 7, 10, 13)
    # local contact positions for each foot
    feet_pos: jp.ndarray = jp.array([[0, 0, -0.02], [0, 0, -0.02], [0, 0, -0.02][0, 0, -0.02]])


class A1Env(MjxEnv):
    def __init__(self, config: A1Config = A1Config()):
        self.config = config

        # Load model.
        model, data = load_mjx_model_from_file(config.model_path)
        super().__init__(
            mj_model=model,
            physics_steps_per_control_step=config.physics_steps_per_control_step,
        )

        self._init_q = self.model.keyframe("standing").qpos
        self._default_ap_pose = self.model.keyframe("standing").qpos[7:]

    def sample_command(self, rng: jax.Array):

