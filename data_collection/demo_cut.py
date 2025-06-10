import pybullet as p
import time
import pickle

import numpy as np

def smooth_data(data, window_size=5):
    """
    Smooth data using a moving average filter.

    Args:
        data (list or np.ndarray): Input data to smooth.
        window_size (int): Size of the moving window.

    Returns:
        np.ndarray: Smoothed data.
    """
    data = np.array(data)
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data, kernel, mode='same')
    return smoothed

def smooth_trajectory(trajectory, window_size=5):
    """
    Apply smoothing to a trajectory (list of lists or 2D array).

    Args:
        trajectory (list of lists or np.ndarray): Input trajectory to smooth.
        window_size (int): Size of the moving window.

    Returns:
        np.ndarray: Smoothed trajectory.
    """
    trajectory = np.array(trajectory)
    smoothed_trajectory = np.array([smooth_data(dim, window_size) for dim in trajectory.T]).T
    return smoothed_trajectory

def main():

    window_size = 100

    fixed_length = 1000
    aligned_demo = {
        'timestamps': [],
        'joint_position': [],
        'gripper_states': [],
        'delta_position': [],
        'delta_end_effector_position': [],
        'delta_end_effector_orientation': []
    }
    policy_name = f'demo/robot_demo_expert2.pkl'
    with open(policy_name, 'rb') as f:
        replay_data = pickle.load(f)
    
    indices = np.linspace(0, len(replay_data['timestamps']) - 1, fixed_length, dtype=int)
    print(indices)
    aligned_demo['timestamps'] = list(range(fixed_length))

    # Smooth and downsample joint positions
    joint_positions = np.array(replay_data['joint_position'])
    smoothed_joint_positions = smooth_trajectory(joint_positions, window_size)
    aligned_demo['joint_position'] = smoothed_joint_positions[indices].tolist()

    # Downsample gripper states (binary, no smoothing needed)
    gripper_states = np.array(replay_data['gripper_states'])
    aligned_demo['gripper_states'] = gripper_states[indices].tolist()

    with open(f'demo/aligned_demo_expert2.pkl', 'wb') as f:
        pickle.dump(aligned_demo, f)


if __name__ == "__main__":
    main()