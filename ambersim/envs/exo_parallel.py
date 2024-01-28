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

    def reset(self, rng: jnp.ndarray, q_init: jnp.ndarray, dq_init: jnp.ndarray, behavState: BehavState) -> State:
        """Resets the vectorized environment."""

        def reset(sys, rng):
            env = self._env_fn(sys=sys)
            return env.reset(rng, q_init, dq_init, behavState)

        state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._sys_v, rng)
        return state

    def barrier_reset(
        self, rng: jnp.ndarray, q_inits: jnp.ndarray, dq_inits: jnp.ndarray, behavState: BehavState
    ) -> State:
        """Resets the vectorized environment."""

        def reset(sys, rng, q_init, dq_init):
            env = self._env_fn(sys=sys)
            return env.reset(rng, q_init, dq_init, behavState)

        state = jax.vmap(reset, in_axes=[self._in_axes, 0, 0, 0])(self._sys_v, rng, q_inits, dq_inits)

        return state

    # def barrier_reset(self, rng: jnp.ndarray, q_inits: jnp.ndarray, dq_inits: jnp.ndarray, behavState: BehavState) -> State:
    #     """Resets the vectorized environment."""

    #     def reset(sys, rng,q_init,dq_init):
    #         env = self._env_fn(sys=sys)
    #         return env.reset(rng, q_init, dq_init, behavState)

    #     vec_env_reset = jax.vmap(reset, in_axes=[self._in_axes, 0,None,None])

    #     state = jax.vmap(vec_env_reset, in_axes=[None, None, None, 0])(self._sys_v, rng,q_inits,dq_inits)

    #     return state

    def barrier_step(self, state: State, action: jnp.ndarray) -> State:
        """Steps the vectorized environment."""

        def step(sys, s, a):
            env = self._env_fn(sys=sys)
            return env.barrier_step(s, a)

        # res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(self._sys_v, state, action)

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
        friction = jax.random.uniform(key, (1,), minval=0.9, maxval=1.1)
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


def random_quaternion(key, max_angle_degrees=10):
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


def random_slope(key, max_angle_degrees=5):
    """Generate a random rotation within a specified maximum angle using JAX."""
    # Generate a random axis (unit vector)
    axis_index = jax.random.randint(key, (1, 1), minval=0, maxval=2)

    # Initialize axis vector
    axis = jnp.zeros(3)
    # Set the chosen axis to 1 (either X or Y)
    axis = axis.at[axis_index].set(1)

    # Generate a random angle within the specified range and convert to radians
    # angle_x = jax.random.uniform(key, minval=-max_angle_degrees, maxval=max_angle_degrees) * jnp.pi / 180
    angle = jax.random.uniform(key, minval=-max_angle_degrees, maxval=max_angle_degrees) * jnp.pi / 180

    # Calculate quaternion components
    s = jnp.sin(angle / 2)
    w = jnp.cos(angle / 2)
    x = axis[0] * s
    y = axis[1] * s
    z = axis[2] * s

    return jnp.array([w, x, y, z])


def updateGeomsQuat(sys, geom_indices, max_degree, rngs):
    """Generate random quaternions for all specified geom_indices."""
    rand_quats = jax.vmap(random_quaternion)(rngs, max_angle_degrees=max_degree)
    geom_quat = sys.geom_quat.at[geom_indices].set(rand_quats)
    return geom_quat


def randomizeSlope(sys, plane_ind, rng, max_angle_degrees=0):
    """Randomizes the plane slope."""
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({"geom_quat": 0})

    # vmap to get multiple copies
    @jax.vmap
    def set_slope(rng):
        rand_quat = random_slope(rng, max_angle_degrees=max_angle_degrees)
        rand_plane_geom = sys.geom_quat.at[plane_ind].set(rand_quat)
        return rand_plane_geom

    rand_geom_quats = set_slope(rng)

    sys_v = sys.tree_replace({"geom_quat": rand_geom_quats})

    return sys_v, in_axes


#   @jax.vmap
#     def get_offset(rng):
#         offset = jax.random.uniform(rng, shape=(3,), minval=-0.1, maxval=0.1)
#         pos = sys.body_pos.at[0].set(offset)
#         return pos

#     sys_v = sys.tree_replace({"body_pos": get_offset(rng)})
#     in_axes = jax.tree_map(lambda x: None, sys)
#     in_axes = in_axes.tree_replace({"body_pos": 0})
#     return sys_v, in_axes


def randomizeSlopeGainWeight(sys, torso_idx, geom_indices, rng, max_angle_degrees, max_gain):
    """Randomizes the plane slope, friction, and gain."""

    @jax.vmap
    def rand(rng):
        _, key = jax.random.split(rng, 2)
        # friction
        friction = jax.random.uniform(key, (1,), minval=0.7, maxval=1.4)
        friction = sys.geom_friction.at[:, geom_indices].set(friction * jnp.ones(len(geom_indices)))

        _, key = jax.random.split(key, 2)
        rand_quat = random_slope(key, max_angle_degrees=max_angle_degrees)
        rand_geom_quats = sys.geom_quat.at[geom_indices[0]].set(rand_quat)

        _, key = jax.random.split(key, 2)
        rand_quat = random_slope(key, max_angle_degrees=max_angle_degrees)
        rand_geom_quats = rand_geom_quats.at[geom_indices[1]].set(rand_quat)

        # actuator
        _, key = jax.random.split(key, 2)
        gain_range = (-max_gain, max_gain)
        param = jax.random.uniform(key, (1,), minval=gain_range[0], maxval=gain_range[1]) + sys.actuator_gainprm[:, 0]
        gain = sys.actuator_gainprm.at[:, 0].set(param)
        bias = sys.actuator_biasprm.at[:, 1].set(-param)
        # #  geom_quat
        # rand_quat = random_slope(rng, max_angle_degrees=max_angle_degrees)
        # rand_plane_geom = sys.geom_quat.at[geom_idx].set(rand_quat)

        com_offset = jax.random.uniform(rng, shape=(3,), minval=-0.05, maxval=0.05) + sys.body_pos[torso_idx]
        pos = sys.body_pos.at[torso_idx].set(com_offset)
        rand_inertia = jax.random.uniform(rng, shape=(3,), minval=-0.01, maxval=0.01) + sys.body_inertia[torso_idx]
        inertia = sys.body_inertia.at[torso_idx].set(rand_inertia)

        return friction, gain, bias, pos, inertia, rand_geom_quats

    friction, gain, bias, pos, inertia, rand_geom_quats = rand(rng)

    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_quat": 0,
            "geom_friction": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
            "body_pos": 0,
            "body_inertia": 0,
        }
    )

    sys = sys.tree_replace(
        {
            "geom_quat": rand_geom_quats,
            "geom_friction": friction,
            "actuator_gainprm": gain,
            "actuator_biasprm": bias,
            "body_pos": pos,
            "body_inertia": inertia,
        }
    )

    return sys, in_axes


def randomizeSlopeGain(sys, plane_ind, rng, max_angle_degrees, max_gain):
    """Randomizes the plane slope, friction, and gain."""

    @jax.vmap
    def rand(rng):
        _, key = jax.random.split(rng, 2)
        # friction
        friction = jax.random.uniform(key, (1,), minval=0.7, maxval=1.4)
        friction = sys.geom_friction.at[:, 0].set(friction)
        # actuator
        _, key = jax.random.split(key, 2)
        gain_range = (-max_gain, max_gain)
        param = jax.random.uniform(key, (1,), minval=gain_range[0], maxval=gain_range[1]) + sys.actuator_gainprm[:, 0]
        gain = sys.actuator_gainprm.at[:, 0].set(param)
        bias = sys.actuator_biasprm.at[:, 1].set(-param)
        #  geom_quat
        rand_quat = random_slope(rng, max_angle_degrees=max_angle_degrees)
        rand_plane_geom = sys.geom_quat.at[plane_ind].set(rand_quat)

        return friction, gain, bias, rand_plane_geom

    friction, gain, bias, rand_plane_geom = rand(rng)

    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_quat": 0,
            "geom_friction": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )

    sys = sys.tree_replace(
        {
            "geom_quat": rand_plane_geom,
            "geom_friction": friction,
            "actuator_gainprm": gain,
            "actuator_biasprm": bias,
        }
    )

    return sys, in_axes


def randomizeBoxTerrain(sys, geom_indices, rng):
    """Randomizes the mjx.Model for geom quat."""
    # Split each of these RNG keys further into 5 parts to get random quaternions for each index
    rngs_for_indices = jax.vmap(lambda rng_key: jax.random.split(rng_key, len(geom_indices)))(rng)

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
