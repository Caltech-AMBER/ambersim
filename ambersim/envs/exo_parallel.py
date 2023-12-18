from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from brax.base import System
from brax.envs.base import Env, PipelineEnv, State, Wrapper
from brax.envs.wrappers import training

from ambersim.envs.exo_base import BehavState, Exo, ExoConfig


class CustomVecEnv(Wrapper):
    """Wrapper for domain randomization."""

    def __init__(
        self,
        env: Env,
        randomization_fn: Callable[[System], Tuple[System, System]],
    ):
        """Initializes the environment."""
        super().__init__(env)
        self._sys_v, self._in_axes = randomization_fn(self.sys)

    def _env_fn(self, sys: System) -> Env:
        """Store the unvectorized env."""
        env = self.env
        env.unwrapped.sys = sys
        return env

    def reset(self, rng: jnp.ndarray, behavState: BehavState) -> State:
        """Resets the vectorized environment."""

        def reset(sys, rng):
            env = self._env_fn(sys=sys)
            return env.reset(rng, behavState)

        state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._sys_v, rng)
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Steps the vectorized environment."""

        def step(sys, s, a):
            env = self._env_fn(sys=sys)
            return env.step(s, a)

        res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(self._sys_v, state, action)
        return res

    def reset_bez(self, rng: jnp.ndarray, alpha, step_dur) -> State:
        """Resets the vectorized environment with bezier polynomial."""

        def reset_bez(sys, rng, alpha, step_dur):
            env = self._env_fn(sys=sys)
            return env.reset_bez(rng, alpha, step_dur)

        state = jax.vmap(reset_bez, in_axes=[self._in_axes, 0, None, None])(self._sys_v, rng, alpha, step_dur)
        return state

    def bez_step(self, state: State, alpha, step_dur) -> State:
        """Steps the vectorized environment with bezier polynomial."""

        def bez_step(sys, s, alpha, step_dur):
            env = self._env_fn(sys=sys)
            return env.bez_step(s, alpha, step_dur)

        res = jax.vmap(bez_step, in_axes=[self._in_axes, 0, None, None])(self._sys_v, state, alpha, step_dur)
        return res


def rand_friction(sys, rng):
    """Randomizes the mjx.Model."""

    @jax.vmap
    def rand(rng):
        _, key = jax.random.split(rng, 2)
        # friction
        friction = jax.random.uniform(key, (1,), minval=0.95, maxval=1.05)
        friction = sys.geom_friction.at[:, 0].set(friction)
        return friction

    friction = rand(rng)

    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({"geom_friction": 0})

    sys = sys.tree_replace({"geom_friction": friction})

    return sys, in_axes


def rand_friction_gain(sys, rng):
    """Randomizes the mjx.Model."""

    @jax.vmap
    def rand(rng):
        _, key = jax.random.split(rng, 2)
        # friction
        friction = jax.random.uniform(key, (1,), minval=0.9, maxval=1.1)
        friction = sys.geom_friction.at[:, 0].set(friction)
        # actuator
        _, key = jax.random.split(key, 2)
        gain_range = (-5, 5)
        param = jax.random.uniform(key, (1,), minval=gain_range[0], maxval=gain_range[1]) + sys.actuator_gainprm[:, 0]
        gain = sys.actuator_gainprm.at[:, 0].set(param)
        bias = sys.actuator_biasprm.at[:, 1].set(-param)
        return friction, gain, bias

    friction, gain, bias = rand(rng)

    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )

    sys = sys.tree_replace(
        {
            "geom_friction": friction,
            "actuator_gainprm": gain,
            "actuator_biasprm": bias,
        }
    )

    return sys, in_axes


def randomizeCoMOffset(sys, rng):
    """Randomizes the mjx.Model for com offset."""

    @jax.vmap
    def get_offset(rng):
        offset = jax.random.uniform(rng, shape=(3,), minval=-0.1, maxval=0.1)
        pos = sys.body_pos.at[0].set(offset)
        return pos

    sys_v = sys.tree_replace({"body_pos": get_offset(rng)})
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({"body_pos": 0})
    return sys_v, in_axes


def random_quaternion(key, max_angle_degrees=30):
    """Generate a random rotation within a specified maximum angle using JAX."""
    # Generate a random axis (unit vector)
    axis = jax.random.normal(key, (3,))
    axis /= jnp.linalg.norm(axis)

    # Generate a random angle within the specified range
    angle = jax.random.uniform(key, minval=-max_angle_degrees, maxval=max_angle_degrees)

    # Convert to radian and create the rotation
    # rotation = jax.scipy.spatial.transform.Rotation.from_rotvec(jnp.array(axis) * jnp.deg2rad(angle))
    # return rotation
    s = jnp.sin(angle / 2)
    w = jnp.cos(angle / 2)
    x = axis[0] * s
    y = axis[1] * s
    z = axis[2] * s
    return jnp.array([w, x, y, z])


def updateGeomsQuat(sys, geom_indices, rngs):
    """Generate random quaternions for all specified geom_indices."""
    rand_quats = jax.vmap(random_quaternion)(rngs)
    geom_quat = sys.geom_quat.at[geom_indices].set(rand_quats)
    return geom_quat


def randomizeBoxTerrain(sys, geom_indices, rngs):
    """Randomizes the mjx.Model for geom quat."""
    # Split each of these RNG keys further into 5 parts to get random quaternions for each index
    rngs_for_indices = jax.vmap(lambda rng_key: jax.random.split(rng_key, len(geom_indices)))(rngs)

    # vmap to get multiple copies
    rand_geom_quats = jax.vmap(updateGeomsQuat, in_axes=(None, None, 0))(sys, geom_indices, rngs_for_indices)

    sys_v = sys.tree_replace({"geom_quat": rand_geom_quats})
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({"geom_quat": 0})
    return sys_v, in_axes


# @dataclass
# class ExoParallelConfig:
#     rand_function: Callable[[System], Tuple[System, System]] = rand_friction


# class ExoParallel(Exo):
#     """Creates an environment from the registry.

#     Args:
#       env_name: environment name string
#       episode_length: length of episode
#       action_repeat: how many repeated actions to take per environment step
#       auto_reset: whether to auto reset the environment after an episode is done
#       batch_size: the number of environments to batch together
#       **kwargs: keyword argments that get passed to the Env class constructor

#     Returns:
#       env: an environment
#     """

#     def __init__(self, config: ExoParallelConfig = ExoParallelConfig, **kwargs):
#         self.base_env = Exo()
#         self.config = config

#         self.randomization_fn = functools.partial(config.rand_function, kwargs["rng"])
#         self.env = VecEnv(self.base_env, self.randomization_fn)


# import functools, jax
# import jax.numpy as jnp
# from typing import Callable, Dict, Optional, Tuple
# from brax.base import System
# from brax.envs.base import Env, PipelineEnv, State, Wrapper
# from brax.envs.wrappers import training
# import utils.terrain_utils as terrain_utils
# import utils.sim_utils as sim_utils


# from envs.exo import Exo
# from envs.hopper import Hopper

# _envs = {
#   'exo': Exo,
#   'hopper':Hopper
# }


# class CustomWrapper(Wrapper):
#   """Wrapper for domain randomization."""

#   def __init__(
#       self,
#       env: Env,
#       randomization_fn: Callable[[System], Tuple[System, System]],
#   ):
#     super().__init__(env)
#     self._sys_v, self._in_axes = randomization_fn(self.sys)

#   def _env_fn(self, sys: System) -> Env:
#     env = self.env
#     env.unwrapped.sys = sys
#     return env

#   def reset(self, rng: jnp.ndarray) -> State:
#     def reset(sys, rng):
#       env = self._env_fn(sys=sys)
#       return env.reset(rng)

#     state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._sys_v, rng)
#     return state

#   def step(self, state: State, action: jnp.ndarray) -> State:
#     def step(sys, s, a):
#       env = self._env_fn(sys=sys)
#       return env.step(s, a)

#     res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
#         self._sys_v, state, action
#     )
#     return res

#   def resetBatch2GivenState(self,des_state,rng):
#     def reset(sys,des_state,rng):
#       env = self._env_fn(sys=sys)
#       return env.reset2GivenStateRand(des_state,rng)

#     state = jax.vmap(reset, in_axes=[self._in_axes, None,0])(self._sys_v, des_state,rng)

#     return state


# def createEnv(
#     env_name: str,
#     episode_length: int = 1000,
#     action_repeat: int = 1,
#     auto_reset: bool = True,
#     batch_size: Optional[int] = None,
#     randomization_fn: Optional[Callable[[System], Tuple[System, System]]
#     ] = None,
#     **kwargs,
# ) -> Env:
#   """Creates an environment from the registry.

#   Args:
#     env_name: environment name string
#     episode_length: length of episode
#     action_repeat: how many repeated actions to take per environment step
#     auto_reset: whether to auto reset the environment after an episode is done
#     batch_size: the number of environments to batch together
#     **kwargs: keyword argments that get passed to the Env class constructor

#   Returns:
#     env: an environment
#   """
#   env = _envs[env_name](**kwargs)

#   # if episode_length is not None:
#   #   env = training.EpisodeWrapper(env, episode_length, action_repeat)
#   if batch_size:
#     if randomization_fn is None:
#         env = training.VmapWrapper(env, batch_size)
#     else:
#       env = CustomWrapper(env, randomization_fn)
#   # if auto_reset:
#     # env = training.AutoResetWrapper(env)

#   return env

# def randomFriction(sys, rng):
#   """Randomizes the mjx.Model."""
#   @jax.vmap
#   def rand(rng):
#     # friction
#     friction = jax.random.uniform(rng, (1,), minval=0.8, maxval=1.2)
#     friction = sys.geom_friction.at[:, 0].set(friction)
#     return friction

#   friction = rand(rng)

#   in_axes = jax.tree_map(lambda x: None, sys)
#   in_axes = in_axes.tree_replace({
#       'geom_friction': 0,
#   })

#   sys = sys.tree_replace({
#       'geom_friction': friction,
#   })

#   return sys, in_axes


# # def randomizeTerrain(sys,rng):
# #   @jax.vmap
# #   def get_offset(rng):
# #     offset = jax.random.uniform(rng, shape=(,), minval=-0.1, maxval=0.1)
# #     pos = sys.body_pos.at[0].set(offset)
# #     return pos

# #   #   return geom_convex_face,geom_convex_vert,geom_convex_edge,geom_convex_facenormal
# #   sys_v  = sys
# #   vert,face,edge,facenormal = jax.vmap(getRandMesh)(rng)
# #   sys_v.geom_convex_face[0] = face
# #   sys_v.geom_convex_vert[0] = vert
# #   # sys_v.geom_convex_edge[0] = edge
# #   # sys_v.geom_convex_facenormal[0] = facenormal
# #   # face,vert,edge,facenormal = update_mesh(rng)
# #   # sys_v = sys.tree_replace({'geom_convex_face':face,'geom_convex_vert':vert})

# #   in_axes = jax.tree_map(lambda x: None, sys)
# #   in_axes = in_axes.tree_replace({'geom_convex_face': 0,
# #                                  'geom_convex_vert':0 })

# #   return sys_v, in_axes
