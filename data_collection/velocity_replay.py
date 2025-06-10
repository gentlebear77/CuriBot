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

# Load the demo data
with open('demo/robot_demo_expert.pkl', 'rb') as f:
    demo_data = pickle.load(f)

# Create a new robot instance
stepsize = 1 / 1000
robot = SimpleEnv(Test_env=True)
robot.Videosave_start(video_name='velocity.mp4')
print(demo_data.keys())
# Get the recorded data from the pickle file
timestamps = demo_data['timestamps']
joint_velocities = demo_data['joint_velocity']
gripper_states = demo_data['gripper_states']


if not os.path.exists('frames'):
    os.makedirs('frames')

initial_position, initial_velocities = robot.getJointStates()
print("Initial joint positions: ", initial_position, "Initial joint velocities: ", initial_velocities)


current_pos, current_ori = robot.getEndEffectorPose()
offset = np.array([0.1, 0.1, 0])
target_pos = current_pos + offset
target_ori = current_ori
target_joint_positions = robot.InverseKinematics(target_pos, current_ori)
robot.setTargetPositions(target_joint_positions)
duration = 3
num_steps = int(duration / stepsize)
for _ in range(20):
    robot.step()
    time.sleep(stepsize)

robot_position, robot_velocities = robot.getJointStates()
print("current_joint_velocities: ", robot_position)

robot.reset_velocities(target_value=robot_position)
robot_position, robot_velocities = robot.getJointStates()

print("joint velocities: ", robot_velocities)
# Initial joint velocities:  [-4.5572218829408577e-17, -2.7881169595289634e-13, 
#                           -8.248214272715774e-17, -2.2102458752115695e-13, -9.713343440495993e-16, 
#                             -1.7352438930196001e-13, 4.765172519382834e-14, 0.0, -3.9404965323005115e-17]


# gif_filename = 'output_video.gif'
# fps = 1200
# duration = 1 / fps  # Duration between frames in seconds
# writer = imageio.get_writer(gif_filename, mode='I', duration=duration)

# Replay the demo
for timestep, joint_velocities, gripper_open in zip(timestamps, joint_velocities, gripper_states):
    if timestep % 1000 == 0:
        print("Simulation time: {:.3f}".format(timestep))
    # Set the recorded joint states for this timestep
    _, robot_velocities = robot.getJointStates()
    # error = np.linalg.norm(np.array(robot_velocities) - np.array(joint_velocities))

    # robot.setTargetPositions(joint_positions)
    robot.setTargetVelocities(joint_velocities)
    # Control the gripper based on recorded gripper state
    robot.controlGripper(gripper=gripper_open)
    # After stepping the simulation
    # joint_pos, joint_vel = robot.getJointStates()
    # print(gripper_open)
    # Step the simulation to the next timestep
    for i in range(10):
        robot.step()

    time.sleep(stepsize)

# writer.close()
robot.Videosave_end()
print("Replay finished.")
