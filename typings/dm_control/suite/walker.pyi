"""
This type stub file was generated by pyright.
"""

from dm_control import mujoco
from dm_control.suite import base

"""Planar Walker Domain."""
_DEFAULT_TIME_LIMIT = ...
_CONTROL_TIMESTEP = ...
_STAND_HEIGHT = ...
_WALK_SPEED = ...
_RUN_SPEED = ...
SUITE = ...

def get_model_and_assets():  # -> tuple[Any, dict[str, Any]]:
    """Returns a tuple containing the model XML string and a dict of assets."""
    ...

@SUITE.add("benchmarking")
def stand(time_limit=..., random=..., environment_kwargs=...):  # -> Environment:
    """Returns the Stand task."""
    ...

@SUITE.add("benchmarking")
def walk(time_limit=..., random=..., environment_kwargs=...):  # -> Environment:
    """Returns the Walk task."""
    ...

@SUITE.add("benchmarking")
def run(time_limit=..., random=..., environment_kwargs=...):  # -> Environment:
    """Returns the Run task."""
    ...

class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Walker domain."""

    def torso_upright(self):
        """Returns projection from z-axes of torso to the z-axes of world."""
        ...
    def torso_height(self):
        """Returns the height of the torso."""
        ...
    def horizontal_velocity(self):
        """Returns the horizontal velocity of the center-of-mass."""
        ...
    def orientations(self):
        """Returns planar orientations of all bodies."""
        ...

class PlanarWalker(base.Task):
    """A planar walker task."""

    def __init__(self, move_speed, random=...) -> None:
        """Initializes an instance of `PlanarWalker`.

        Args:
          move_speed: A float. If this value is zero, reward is given simply for
            standing up. Otherwise this specifies a target horizontal velocity for
            the walking task.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        ...
    def initialize_episode(self, physics):  # -> None:
        """Sets the state of the environment at the start of each episode.

        In 'standing' mode, use initial orientation and small velocities.
        In 'random' mode, randomize joint angles and let fall to the floor.

        Args:
          physics: An instance of `Physics`.

        """
        ...
    def get_observation(self, physics):  # -> OrderedDict[Unknown, Unknown]:
        """Returns an observation of body orientations, height and velocites."""
        ...
    def get_reward(self, physics):
        """Returns a reward to the agent."""
        ...
