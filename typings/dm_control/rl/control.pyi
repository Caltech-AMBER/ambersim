"""
This type stub file was generated by pyright.
"""

import abc
import contextlib

import dm_env

"""
This type stub file was generated by pyright.
"""
FLAT_OBSERVATION_KEY = ...

class Environment(dm_env.Environment):
    """Class for physics-based reinforcement learning environments."""

    def __init__(
        self,
        physics,
        task,
        time_limit=...,
        control_timestep=...,
        n_sub_steps=...,
        flat_observation=...,
        legacy_step: bool = ...,
    ) -> None:
        """Initializes a new `Environment`.

        Args:
          physics: Instance of `Physics`.
          task: Instance of `Task`.
          time_limit: Optional `int`, maximum time for each episode in seconds. By
            default this is set to infinite.
          control_timestep: Optional control time-step, in seconds.
          n_sub_steps: Optional number of physical time-steps in one control
            time-step, aka "action repeats". Can only be supplied if
            `control_timestep` is not specified.
          flat_observation: If True, observations will be flattened and concatenated
            into a single numpy array.
          legacy_step: If True, steps the state with up-to-date position and
            velocity dependent fields. See Page 6 of
            https://arxiv.org/abs/2006.12983 for more information.

        Raises:
          ValueError: If both `n_sub_steps` and `control_timestep` are supplied.
        """
        ...
    def reset(self):
        """Starts a new episode and returns the first `TimeStep`."""
        ...
    def step(self, action):
        """Updates the environment using the action and returns a `TimeStep`."""
        ...
    def action_spec(self):
        """Returns the action specification for this environment."""
        ...
    def step_spec(self):
        """May return a specification for the values returned by `step`."""
        ...
    def observation_spec(self):
        """Returns the observation specification for this environment.

        Infers the spec from the observation, unless the Task implements the
        `observation_spec` method.

        Returns:
          An dict mapping observation name to `ArraySpec` containing observation
          shape and dtype.
        """
        ...
    @property
    def physics(self): ...
    @property
    def task(self): ...
    def control_timestep(self):
        """Returns the interval between agent actions in seconds."""
        ...

def compute_n_steps(control_timestep, physics_timestep, tolerance=...):
    """Returns the number of physics timesteps in a single control timestep.

    Args:
      control_timestep: Control time-step, should be an integer multiple of the
        physics timestep.
      physics_timestep: The time-step of the physics simulation.
      tolerance: Optional tolerance value for checking if `physics_timestep`
        divides `control_timestep`.

    Returns:
      The number of physics timesteps in a single control timestep.

    Raises:
      ValueError: If `control_timestep` is smaller than `physics_timestep` or if
        `control_timestep` is not an integer multiple of `physics_timestep`.
    """
    ...

class Physics(metaclass=abc.ABCMeta):
    """Simulates a physical environment."""

    legacy_step: bool = ...
    @abc.abstractmethod
    def step(self, n_sub_steps=...):
        """Updates the simulation state.

        Args:
          n_sub_steps: Optional number of times to repeatedly update the simulation
            state. Defaults to 1.
        """
        ...
    @abc.abstractmethod
    def time(self):
        """Returns the elapsed simulation time in seconds."""
        ...
    @abc.abstractmethod
    def timestep(self):
        """Returns the simulation timestep."""
        ...
    def set_control(self, control):
        """Sets the control signal for the actuators."""
        ...
    @contextlib.contextmanager
    def reset_context(self):
        """Context manager for resetting the simulation state.

        Sets the internal simulation to a default state when entering the block.

        ```python
        with physics.reset_context():
          # Set joint and object positions.

        physics.step()
        ```

        Yields:
          The `Physics` instance.
        """
        ...
    @abc.abstractmethod
    def reset(self):
        """Resets internal variables of the physics simulation."""
        ...
    @abc.abstractmethod
    def after_reset(self):
        """Runs after resetting internal variables of the physics simulation."""
        ...
    def check_divergence(self):
        """Raises a `PhysicsError` if the simulation state is divergent.

        The default implementation is a no-op.
        """
        ...

class PhysicsError(RuntimeError):
    """Raised if the state of the physics simulation becomes divergent."""

    ...

class Task(metaclass=abc.ABCMeta):
    """Defines a task in a `control.Environment`."""

    @abc.abstractmethod
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Called by `control.Environment` at the start of each episode *within*
        `physics.reset_context()` (see the documentation for `base.Physics`).

        Args:
          physics: Instance of `Physics`.
        """
        ...
    @abc.abstractmethod
    def before_step(self, action, physics):
        """Updates the task from the provided action.

        Called by `control.Environment` before stepping the physics engine.

        Args:
          action: numpy array or array-like action values, or a nested structure of
            such arrays. Should conform to the specification returned by
            `self.action_spec(physics)`.
          physics: Instance of `Physics`.
        """
        ...
    def after_step(self, physics):
        """Optional method to update the task after the physics engine has stepped.

        Called by `control.Environment` after stepping the physics engine and before
        `control.Environment` calls `get_observation, `get_reward` and
        `get_termination`.

        The default implementation is a no-op.

        Args:
          physics: Instance of `Physics`.
        """
        ...
    @abc.abstractmethod
    def action_spec(self, physics):
        """Returns a specification describing the valid actions for this task.

        Args:
          physics: Instance of `Physics`.

        Returns:
          A `BoundedArraySpec`, or a nested structure containing `BoundedArraySpec`s
          that describe the shapes, dtypes and elementwise lower and upper bounds
          for the action array(s) passed to `self.step`.
        """
        ...
    def step_spec(self, physics):
        """Returns a specification describing the time_step for this task.

        Args:
          physics: Instance of `Physics`.

        Returns:
          A `BoundedArraySpec`, or a nested structure containing `BoundedArraySpec`s
          that describe the shapes, dtypes and elementwise lower and upper bounds
          for the array(s) returned by `self.step`.
        """
        ...
    @abc.abstractmethod
    def get_observation(self, physics):
        """Returns an observation from the environment.

        Args:
          physics: Instance of `Physics`.
        """
        ...
    @abc.abstractmethod
    def get_reward(self, physics):
        """Returns a reward from the environment.

        Args:
          physics: Instance of `Physics`.
        """
        ...
    def get_termination(self, physics):
        """If the episode should end, returns a final discount, otherwise None."""
        ...
    def observation_spec(self, physics):
        """Optional method that returns the observation spec.

        If not implemented, the Environment infers the spec from the observation.

        Args:
          physics: Instance of `Physics`.

        Returns:
          A dict mapping observation name to `ArraySpec` containing observation
          shape and dtype.
        """
        ...

def flatten_observation(observation, output_key=...):
    """Flattens multiple observation arrays into a single numpy array.

    Args:
      observation: A mutable mapping from observation names to numpy arrays.
      output_key: The key for the flattened observation array in the output.

    Returns:
      A mutable mapping of the same type as `observation`. This will contain a
      single key-value pair consisting of `output_key` and the flattened
      and concatenated observation array.

    Raises:
      ValueError: If `observation` is not a `collections.abc.MutableMapping`.
    """
    ...
