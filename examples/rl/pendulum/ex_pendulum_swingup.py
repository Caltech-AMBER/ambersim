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
        num_timesteps=1_000_000,
        num_evals=10,
        reward_scaling=1,
        episode_length=10000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=8,
        gae_lambda=0.95,
        num_updates_per_batch=4,
        discounting=0.99,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=8192,
        batch_size=1024,
        network_factory=make_networks_factory,
        num_resets_per_eval=10,
        seed=0,
    )

    def progress(num_steps, metrics):
        """Logs progress during RL."""
        # times.append(datetime.now())
        # x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        # ydataerr.append(metrics['eval/episode_reward_std'])

        # plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])

        # plt.xlabel('# environment steps')
        # plt.ylabel('reward per episode')
        # plt.title(f'y={y_data[-1]:.3f}')

        # plt.errorbar(x_data, y_data, yerr=ydataerr)
        # plt.show()
        print(y_data[-1])

    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]

    env = envs.get_environment(env_name)
    eval_env = envs.get_environment(env_name)
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
    )

    # #### #
    # eval #
    # #### #
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

    eval_env = envs.get_environment(env_name)

    jit_reset = jit(eval_env.reset)
    jit_step = jit(eval_env.step)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    rollout = [state]
    images = [get_image(state, camera="camera")]

    # grab a trajectory
    n_steps = 500
    render_every = 2

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)
        if i % render_every == 0:
            images.append(get_image(state, camera="camera"))

        if state.done:
            break
    # media.show_video(images, fps=1.0 / eval_env.dt / render_every)

    # saving images as video
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
    breakpoint()
