import sys
sys.path.append('../src')
import os
import cv2
import pybullet as p
import time
import numpy as np
import pickle
from dinobot.env.simple import SimpleEnv
import imageio
from scipy.ndimage import gaussian_filter


# Load the demo data
with open('demo/robot_demo_1.pkl', 'rb') as f:
    demo_data = pickle.load(f)
print(demo_data.keys())
# Create a new robot instance
stepsize = 1 / 100
robot = SimpleEnv(Test_env=False)
time.sleep(1)
# robot.offset()
robot.reset()
# Get the recorded data from the pickle file
timestamps = demo_data['timestamps']
# joint_positions = demo_data['joint_position']
joint_positions = demo_data['joint_states']
# joint_velocities = demo_data['joint_velocity']
gripper_states = demo_data['gripper_states']

deltaposition_control = {
    'timestamps': [],
    'joint_position': [],
    'gripper_states': [],
    'joint_velocity': [],
    'delta_position': [],
    'delta_end_effector_position': [],
    'delta_end_effector_orientation': []
}

frame_count = 0

robot_position, robot_velocities = robot.getJointStates()
print("Initial joint velocities: ", robot_velocities)
#Initial joint velocities:  [7.408156197476936e-11, -1.7612949013753182e-06, 
#                            9.486269003903524e-10, -8.077674093123766e-06, 4.665799412164759e-09, 
#                            -5.464378949326942e-15, -7.515528134250839e-12, 0.0, 3.764436772794202e-10]

# joint_positions = gaussian_filter(joint_positions, sigma=2)

max_velocity = 0
previous_joint_positions, _ = robot.getJointStates()
previous_end_effector_position, previous_end_effector_oritentation = robot.getEndEffectorPose()
# Replay the demo
for timestep, joint_positions, gripper_open in zip(timestamps, joint_positions, gripper_states):
    if timestep % 1000 == 0:
        print("Simulation time: {:.3f}".format(timestep))
    # Set the recorded joint states for this timestep
    
    robot.setTargetPositions(joint_positions)
    # velocity_calculated = (np.array(joint_positions) - np.array(previous_joint_positions)) / np.array([stepsize])
    
    
    # robot.setTargetVelocities(joint_velocities)
    # robot.setDeltaPositionControl(delta_position, previous_joint_positions)
    # _, robot_velocities = robot.getJointStates()
    # error = np.linalg.norm(np.array(joint_positions) - np.array(previous_joint_positions + delta_position))
    # if error:
    #     print(error)

    robot.controlGripper(gripper=gripper_open)

    for i in range(10):
        robot.step()


    joint_positions, _ = robot.getJointStates()
    delta_position = np.array(joint_positions) - np.array(previous_joint_positions)
    previous_joint_positions = joint_positions

    end_effector_position, end_effector_oritentation = robot.getEndEffectorPose()
    delta_end_effector_position = np.array(end_effector_position) - np.array(previous_end_effector_position)
    delta_end_effector_oritentation = np.array(end_effector_oritentation) - np.array(previous_end_effector_oritentation)
    previous_end_effector_position = end_effector_position
    previous_end_effector_oritentation = end_effector_oritentation

    print("end_effector_position: ", end_effector_position)
    # Record the data
    deltaposition_control['timestamps'].append(timestep)
    deltaposition_control['joint_position'].append(joint_positions)
    deltaposition_control['gripper_states'].append(gripper_open)
    deltaposition_control['delta_position'].append(delta_position)
    # print("delta_position: ", delta_position)
    deltaposition_control['delta_end_effector_position'].append(delta_end_effector_position)
    deltaposition_control['delta_end_effector_orientation'].append(delta_end_effector_oritentation)

    # print("delta_end_effector_position: ", delta_end_effector_position)
    # actual_joint_positions, _ = robot.getJointStates()
    # error = np.linalg.norm(np.array(joint_positions) - actual_joint_positions)
    # if error > 1e-3:
    #     print(f"Error at timestep {timestep}: {error:.6f}")
    
    time.sleep(stepsize)

print("Max velocity: ", max_velocity)
# writer.close()
with open('demo/robot_delta_3.pkl', 'wb') as f:
    pickle.dump(deltaposition_control, f)

print("Replay finished.")
