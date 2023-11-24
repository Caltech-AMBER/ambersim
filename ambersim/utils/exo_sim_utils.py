# helper function that loop through rollout and extract relevatn information
# get control action
# check termination status;
# determine action scale based on joint limit
# observation space: com position, contact force, joint position;


import matplotlib.pyplot as plt
import numpy as np


def plot_tracking_error(env, rollout, actions):
    """Plot the tracking error for each joint or action over a rollout in subplots.

    Args:
        env: The environment object.
        rollout (list): The rollout data, where each item contains state and info.
        actions (list): List of actions taken during the rollout.
    """
    # Extract actual values, nominal actions, and control values
    actual_values = [step.pipeline_state.qpos[-12:] for step in rollout]
    actual_velocities = [step.pipeline_state.qvel[-12:] for step in rollout]
    nominal_actions = [step.info["nominal_action"] for step in rollout]
    ctrl_values = [env.conv_action_based_on_idx(action) for action in actions]

    # Compute target values
    target_values = [nominal + ctrl for nominal, ctrl in zip(nominal_actions, ctrl_values)]

    # Compute tracking error for each joint or action
    tracking_errors = [
        [target[i] - actual[i] for target, actual in zip(target_values, actual_values)]
        for i in range(len(actual_values[0]))
    ]

    # Determine the number of subplots
    num_joints = len(actual_values[0])
    num_cols = 3  # You can adjust this based on your preference
    num_rows = (num_joints + num_cols - 1) // num_cols

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Plot tracking error for each joint in a separate subplot
    for i in range(num_joints):
        axes[i].plot(tracking_errors[i], label=f"Joint {i+1} Tracking Error")
        axes[i].set_title(f"Joint {i+1} Tracking Error")
        axes[i].set_xlabel("Step")
        axes[i].set_ylabel("Error")
        axes[i].legend()
        axes[i].grid(True)

    # Hide unused subplots
    for i in range(num_joints, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Plot tracking error for each joint in a separate subplot
    for i in range(num_joints):
        axes[i].plot(actual_velocities[i], label=f"Joint {i+1} Velocity")
        axes[i].set_title(f"Joint {i+1} Velocity")
        axes[i].set_xlabel("Step")
        axes[i].set_ylabel("Error")
        axes[i].legend()
        axes[i].grid(True)

    # Hide unused subplots
    for i in range(num_joints, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def plot_rewards(rollout):
    """Plot reward tuple during rollout.

    Args:
        rollout (list): The rollout data, where each item contains state and info.
    """
    # Initialize lists to store reward components
    forward_rewards, healthy_rewards, ctrl_costs = [], [], []
    tracking_lin_vel_rewards, tracking_ang_vel_rewards = [], []
    lin_vel_z_penalties, ang_vel_xy_penalties, cop_penalties, action_rate_penalties = [], [], [], []
    total_rewards = []

    # Extract and store each reward component from the rollout
    for state in rollout:
        reward_info = state.info["reward_tuple"]  # Assuming this is how you access the reward information
        forward_rewards.append(reward_info["forward_reward"])
        healthy_rewards.append(reward_info["healthy_reward"])
        ctrl_costs.append(reward_info["ctrl_cost"])
        tracking_lin_vel_rewards.append(reward_info["tracking_lin_vel_reward"])
        tracking_ang_vel_rewards.append(reward_info["tracking_ang_vel_reward"])
        lin_vel_z_penalties.append(reward_info["lin_vel_z_penalty"])
        ang_vel_xy_penalties.append(reward_info["ang_vel_xy_penalty"])
        cop_penalties.append(reward_info["cop_penalty"])
        action_rate_penalties.append(reward_info["action_rate_penalty"])
        total_rewards.append(reward_info["total_reward"])

    # Convert lists to numpy arrays for easier handling
    # Now create subplots for each reward component
    plt.figure(figsize=(15, 10))
    time_steps = np.arange(len(rollout))

    reward_components = [
        forward_rewards,
        healthy_rewards,
        ctrl_costs,
        tracking_lin_vel_rewards,
        tracking_ang_vel_rewards,
        lin_vel_z_penalties,
        ang_vel_xy_penalties,
        cop_penalties,
        action_rate_penalties,
        total_rewards,
    ]
    reward_labels = [
        "Forward Reward",
        "Healthy Reward",
        "Control Cost",
        "Tracking Linear Velocity Reward",
        "Tracking Angular Velocity Reward",
        "Linear Velocity Z Penalty",
        "Angular Velocity XY Penalty",
        "CoP Penalty",
        "Action Rate Penalty",
        "Total Reward",
    ]

    for i, (rewards, label) in enumerate(zip(reward_components, reward_labels)):
        plt.subplot(5, 2, i + 1)
        plt.plot(time_steps, rewards)
        plt.title(label)
        plt.xlabel("Time Step")
        plt.ylabel("Value")

    plt.tight_layout()
    plt.show()


def plot_rollout(self, rollout):
    """Plot states info during rollout."""
    # Assuming the lengths of each observation field based on your description
    pos_len, vel_len, contact_force_len, nominal_action_len, last_action_len = (
        self.model.nq - 3,
        self.model.nv - 3,
        8,
        self.model.nu,
        self.custom_act_space_size,
    )

    # Initialize lists to store separated fields
    positions, velocities, contact_forces, nominal_actions, last_actions = [], [], [], [], []

    # Extract and store each field from the rollout
    for state in rollout:
        obs = state.obs  # Assuming this is how you access the observation history
        current_idx = 0
        positions.append(obs[current_idx : current_idx + pos_len])
        current_idx += pos_len
        velocities.append(obs[current_idx : current_idx + vel_len])
        current_idx += vel_len
        contact_forces.append(obs[current_idx : current_idx + contact_force_len])
        current_idx += contact_force_len
        nominal_actions.append(obs[-(nominal_action_len + last_action_len) : -last_action_len])
        last_actions.append(obs[-last_action_len:])

    # Convert lists to numpy arrays for easier handling
    positions = np.array(positions)
    velocities = np.array(velocities)
    contact_forces = np.array(contact_forces)
    nominal_actions = np.array(nominal_actions)
    last_actions = np.array(last_actions)

    # Function to create subplots for a given field in a 3-column format
    def create_subplots(field, field_name, dimensions):
        time_steps = np.arange(len(rollout))
        num_rows = int(np.ceil(dimensions / 3))
        plt.figure(figsize=(15, num_rows * 3))
        for i in range(dimensions):
            plt.subplot(num_rows, 3, i + 1)
            plt.plot(time_steps, field[:, i])
            plt.title(f"{field_name} Dimension {i + 1}")
            plt.xlabel("Time Step")
            plt.ylabel(f"{field_name} Value")
        plt.tight_layout()
        plt.show()

    # Create subplots for each field
    create_subplots(positions, "Position", pos_len)
    create_subplots(velocities, "Velocity", vel_len)
    create_subplots(contact_forces, "Contact Force", contact_force_len)
    create_subplots(nominal_actions, "Nominal Action", nominal_action_len)
    create_subplots(last_actions, "Last Action", last_action_len)
