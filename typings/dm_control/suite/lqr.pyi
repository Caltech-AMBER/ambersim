"""
This type stub file was generated by pyright.
"""

from dm_control import mujoco
from dm_control.suite import base

"""Procedurally generated LQR domain."""
_DEFAULT_TIME_LIMIT = ...
_CONTROL_COST_COEF = ...
SUITE = ...

def get_model_and_assets(n_bodies, n_actuators, random):  # -> tuple[Unknown, dict[str, Any]]:
    """Returns the model description as an XML string and a dict of assets.

    Args:
      n_bodies: An int, number of bodies of the LQR.
      n_actuators: An int, number of actuated bodies of the LQR. `n_actuators`
        should be less or equal than `n_bodies`.
      random: A `numpy.random.RandomState` instance.

    Returns:
      A tuple `(model_xml_string, assets)`, where `assets` is a dict consisting of
      `{filename: contents_string}` pairs.
    """
    ...

@SUITE.add()
def lqr_2_1(time_limit=..., random=..., environment_kwargs=...):  # -> Environment:
    """Returns an LQR environment with 2 bodies of which the first is actuated."""
    ...

@SUITE.add()
def lqr_6_2(time_limit=..., random=..., environment_kwargs=...):  # -> Environment:
    """Returns an LQR environment with 6 bodies of which first 2 are actuated."""
    ...

class Physics(mujoco.Physics):
    """Physics simulation with additional features for the LQR domain."""

    def state_norm(self):  # -> floating[Any]:
        """Returns the norm of the physics state."""
        ...

class LQRLevel(base.Task):
    """A Linear Quadratic Regulator `Task`."""

    _TERMINAL_TOL = ...
    def __init__(self, control_cost_coef, random=...) -> None:
        """Initializes an LQR level with cost = sum(states^2) + c*sum(controls^2).

        Args:
          control_cost_coef: The coefficient of the control cost.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).

        Raises:
          ValueError: If the control cost coefficient is not positive.
        """
        ...
    @property
    def control_cost_coef(self):  # -> Unknown:
        ...
    def initialize_episode(self, physics):  # -> None:
        """Random state sampled from a unit sphere."""
        ...
    def get_observation(self, physics):  # -> OrderedDict[Unknown, Unknown]:
        """Returns an observation of the state."""
        ...
    def get_reward(self, physics):  # -> Any:
        """Returns a quadratic state and control reward."""
        ...
    def get_evaluation(self, physics):  # -> float:
        """Returns a sparse evaluation reward that is not used for learning."""
        ...
    def get_termination(self, physics):  # -> float | None:
        """Terminates when the state norm is smaller than epsilon."""
        ...
