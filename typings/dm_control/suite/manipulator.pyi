"""
This type stub file was generated by pyright.
"""

from dm_control import mujoco
from dm_control.suite import base

"""Planar Manipulator domain."""
_CLOSE = ...
_CONTROL_TIMESTEP = ...
_TIME_LIMIT = ...
_P_IN_HAND = ...
_P_IN_TARGET = ...
_ARM_JOINTS = ...
_ALL_PROPS = ...
_TOUCH_SENSORS = ...
SUITE = ...

def make_model(use_peg, insert):  # -> tuple[Unknown, dict[str, Any]]:
    """Returns a tuple containing the model XML string and a dict of assets."""
    ...

@SUITE.add("benchmarking", "hard")
def bring_ball(fully_observable=..., time_limit=..., random=..., environment_kwargs=...):  # -> Environment:
    """Returns manipulator bring task with the ball prop."""
    ...

@SUITE.add("hard")
def bring_peg(fully_observable=..., time_limit=..., random=..., environment_kwargs=...):  # -> Environment:
    """Returns manipulator bring task with the peg prop."""
    ...

@SUITE.add("hard")
def insert_ball(fully_observable=..., time_limit=..., random=..., environment_kwargs=...):  # -> Environment:
    """Returns manipulator insert task with the ball prop."""
    ...

@SUITE.add("hard")
def insert_peg(fully_observable=..., time_limit=..., random=..., environment_kwargs=...):  # -> Environment:
    """Returns manipulator insert task with the peg prop."""
    ...

class Physics(mujoco.Physics):
    """Physics with additional features for the Planar Manipulator domain."""

    def bounded_joint_pos(self, joint_names):  # -> NDArray[Any]:
        """Returns joint positions as (sin, cos) values."""
        ...
    def joint_vel(self, joint_names):
        """Returns joint velocities."""
        ...
    def body_2d_pose(self, body_names, orientation=...):  # -> NDArray[Unknown]:
        """Returns positions and/or orientations of bodies."""
        ...
    def touch(self):  # -> Any:
        ...
    def site_distance(self, site1, site2):  # -> floating[Any]:
        ...

class Bring(base.Task):
    """A Bring `Task`: bring the prop to the target."""

    def __init__(self, use_peg, insert, fully_observable, random=...) -> None:
        """Initialize an instance of the `Bring` task.

        Args:
          use_peg: A `bool`, whether to replace the ball prop with the peg prop.
          insert: A `bool`, whether to insert the prop in a receptacle.
          fully_observable: A `bool`, whether the observation should contain the
            position and velocity of the object being manipulated and the target
            location.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        ...
    def initialize_episode(self, physics):  # -> None:
        """Sets the state of the environment at the start of each episode."""
        ...
    def get_observation(self, physics):  # -> OrderedDict[Unknown, Unknown]:
        """Returns either features or only sensors (to be used with pixels)."""
        ...
    def get_reward(self, physics):  # -> float | NDArray[Any]:
        """Returns a reward to the agent."""
        ...
