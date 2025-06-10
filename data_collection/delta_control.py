import sys
sys.path.append('../src')
import os
import cv2
import pybullet as p
import time
import numpy as np
import pickle
from dinobot.env.base import SimpleEnv
import imageio
from scipy.ndimage import gaussian_filter


# Load the demo data
with open('demo/robot_demo_delta_position.pkl', 'rb') as f:
    demo_data = pickle.load(f)
# Create a new robot instance
print(demo_data.keys())
stepsize = 1 / 1000
robot = SimpleEnv(Test_env=False)
time.sleep(1)
# robot.offset()
# robot.reset()
# Get the recorded data from the pickle file
timestamps = demo_data['timestamps']
joint_positions = demo_data['joint_position']
joint_velocities = demo_data['joint_velocity']
gripper_states = demo_data['gripper_states']
delta_effector_positions = demo_data['delta_effector_positions']
# delta_positions = demo_data['delta_position']
delta_end_effector_position = demo_data['delta_end_effector_position']
delta_end_effector_orientation = demo_data['delta_end_effector_orientation']
deltaposition_control = {
    'timestamps': [],
    'joint_position': [],
    'gripper_states': [],
    'joint_velocity': [],
    'delta_position': [],
    'delta_end_effector': []
}

replay_end_position = []


def set_offset():
    current_pos, current_ori = robot.getEndEffectorPose()
    print("current_pos: ", current_pos)
    offset = np.array([0.1, 0.1, 0])
    target_pos = current_pos + offset
    target_ori = current_ori
    print("target_pos: ", target_pos)
    robot.setEndEffectorPose(target_pos, target_ori)
    # target_joint_positions = robot.InverseKinematics(target_pos, current_ori)
    # robot.setTargetPositions(target_joint_positions)
    for _ in range(30):
        robot.step()
        time.sleep(stepsize)
    current_pos, current_ori = robot.getEndEffectorPose()
    print("extra_pos: ", current_pos)


def replay():
    # Replay the demo
    print("Replaying the demo...")
    print("demo_length: ", len(timestamps))
    gripper_state = False
    # print("gripper_state: ", gripper_state)
    for timestep, joint_position, joint_velocitiy, gripper_open , delta_end_position, end_position, end_oritentation in zip(timestamps, 
                                                                                                        joint_positions, 
                                                                                                        joint_velocities, 
                                                                                                        gripper_states,
                                                                                                        delta_effector_positions,
                                                                                                        delta_end_effector_position, 
                                                                                                        delta_end_effector_orientation):

        print("Simulation time: {:.3f}".format(timestep))
        # Set the recorded joint states for this timestep
        print("(end_position: ", end_position)
        robot.setDeltaEndControl(end_position, end_oritentation)
        # robot.setDeltaPositionControl(end_position)
        # robot.setDeltaPositionControl(delta_position)\
        # print("joint_velocities: ", joint_velocities)
        # robot.setTargetVelocities(joint_velocitiy)
        if gripper_open != gripper_state:
            gripper_state = gripper_open
            robot.controlGripper(gripper=gripper_open)

        for i in range(10):
            robot.step()        
        # print("current_pos: ", current_pos)
        # replay_end_position.append(current_pos)

        # joint_positions, _ = robot.getJointStates()
        # delta_position = np.array(joint_positions) - np.array(previous_joint_positions)
        # previous_joint_positions = joint_positions

        # end_effector = robot.getEndEffectorPose()
        # delta_end_effector = np.array(end_effector) - np.array(previous_end_effector)
        # previous_end_effector = end_effector

        # # Record the data
        # deltaposition_control['timestamps'].append(timestep)
        # deltaposition_control['joint_position'].append(joint_position)
        # deltaposition_control['joint_velocity'].append(joint_velocities)
        # deltaposition_control['gripper_states'].append(gripper_open)
        # deltaposition_control['delta_position'].append(delta_position)
        # deltaposition_control['delta_end_effector'].append(robot.getEndEffectorPose())
        # actual_joint_positions, _ = robot.getJointStates()
        # error = np.linalg.norm(np.array(joint_positions) - actual_joint_positions)
        # if error > 1e-3:
        #     print(f"Error at timestep {timestep}: {error:.6f}")
        
        time.sleep(stepsize)
    return replay_end_position

if __name__ == '__main__':
    # set_offset()
    robot.Videosave_start(video_name='demo')
    replay()
    robot.Videosave_end()
    trajectory = np.array(replay_end_position)
    replay_end_position = []
    # robot.reset()
    # set_offset()
    # trajec2 = replay()
    current_pos, current_ori = robot.getEndEffectorPose()
    print("current_pos: ", current_pos)
    # writer.close()
    # with open('demo/robot_delta_0.pkl', 'wb') as f:
    #     pickle.dump(deltaposition_control, f)

    print("Replay finished.")
