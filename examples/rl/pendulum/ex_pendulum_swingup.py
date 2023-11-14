import functools
from datetime import datetime

import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from brax import envs
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from jax import jit
from mujoco import mjx

from ambersim.rl.base import State
from ambersim.rl.pendulum.swingup import PendulumSwingupEnv

if __name__ == "__main__":
    # initialize the env
    env_name = "pendulum_swingup"
    envs.register_environment("pendulum_swingup", PendulumSwingupEnv)
    env = envs.get_environment(env_name)

    # boilerplate PPO stuff that we can probably write utils to reduce
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128),
    )
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=100_000,
        num_evals=50,
        reward_scaling=0.1,
        episode_length=200,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=0,
        num_envs=1024,
        batch_size=512,
        network_factory=make_networks_factory,
        seed=0,
    )

    times = [datetime.now()]

    def progress(num_steps, metrics):
        """Logs progress during RL."""
        print(f'num_steps={num_steps}, reward={metrics["eval/episode_reward"]}')
        times.append(datetime.now())

    # Do the training
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
    )

    print(f"Time to jit: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

    # Render a trajectory with the trained policy
    renderer = mj.Renderer(env.model)

    def get_image(state: State, camera: str) -> np.ndarray:
        """Renders the environment state."""
        d = mj.MjData(env.model)
        mjx.device_get_into(d, state.pipeline_state)
        mj.mj_forward(env.model, d)
        renderer.update_scene(d, camera=camera)
        return renderer.render()

    inference_fn = make_inference_fn(params)
    jit_inference_fn = jit(inference_fn)

    jit_reset = jit(env.reset)
    jit_step = jit(env.step)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    images = [get_image(state, camera="camera")]

    # grab a trajectory
    n_steps = 500
    render_every = 2

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        if i % render_every == 0:
            images.append(get_image(state, camera="camera"))

        if state.done:
            break

    # Save images as video
    output_video_file = "output_video.mp4"
    height, width, channels = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can try 'XVID' or 'MJPG' as well
    video_writer = cv2.VideoWriter(output_video_file, fourcc, 10.0, (width, height))

    # Iterate through the list of images and write each frame to the video file
    for image in images:
        # Convert the byte array to a NumPy array
        frame = np.frombuffer(image, dtype=np.uint8).reshape(height, width, channels)

        # Write the frame to the video file
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()
