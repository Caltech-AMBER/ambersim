import functools
import time
from datetime import datetime

import jax
import mujoco
import mujoco.viewer
from brax import envs
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco import mjx

from ambersim.rl.pendulum.swingup import PendulumSwingupEnv

if __name__ == "__main__":
    # Initialize the environment
    envs.register_environment("pendulum_swingup", PendulumSwingupEnv)
    env = envs.get_environment("pendulum_swingup")

    # Define the training function
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(64,) * 3,
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
        network_factory=network_factory,
        seed=0,
    )

    # Define a callback to log progress
    times = [datetime.now()]

    def progress(num_steps, metrics):
        """Logs progress during RL."""
        print(f"  Steps: {num_steps}, Reward: {metrics['eval/episode_reward']}")
        times.append(datetime.now())

    # Do the training
    print("Training...")
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
    )

    print(f"Time to jit: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

    # Run an interactive simulation with the trained policy
    mj_model = env.model
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[0] = 0.1  # initial angle
    obs = env.compute_obs(mjx.device_put(mj_data), {})

    policy = make_inference_fn(params)
    jit_policy = jax.jit(policy)

    rng = jax.random.PRNGKey(0)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            start_time = time.time()
            act_rng, rng = jax.random.split(rng)

            # Apply the policy
            act, _ = jit_policy(obs, act_rng)
            mj_data.ctrl[:] = act
            obs = env.compute_obs(mjx.device_put(mj_data), {})

            # Step the simulation
            for _ in range(env._physics_steps_per_control_step):
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            dt = float(env.dt)
            if elapsed < dt:
                time.sleep(dt - elapsed)
