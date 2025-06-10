import sys
# sys.path.append('../src')

import pybullet as p
import time
from math import sin
import cv2
import numpy as np
# import matplotlib as plt
import pickle
from dinobot.env.simple import SimpleEnv

# import matplotlib.pyplot as plt
duration = 3000
stepsize = 1 / 1000

robot = SimpleEnv(Test_env=False)
# robot.offset()

ee_pos, _ = list(robot.getEndEffectorPose())
ee_pos = list(ee_pos)
previous_end_effector = ee_pos
    # print(ee_pos)
key_map = {
    ord(','): [0, 0.01, 0],  # Move forward in Y
    ord('.'): [0, -0.01, 0], # Move backward in Y
    ord('a'): [-0.01, 0, 0], # Move left in X
    ord('d'): [0.01, 0, 0],  # Move right in X
    ord('q'): [0, 0, 0.01],  # Move up in Z
    ord('e'): [0, 0, -0.01],  # Move down in Z
    ord('r'): [0, 0, 0],  # Reset
    ord('s'): 'stop',  # Stop
    ord('o'): 'open_gripper', # Open gripper
    ord('c'): 'close_gripper' # Close gripper
}
demo_data = {
    'timestamps': [],
    'delta_effector_positions': [],
    'joint_position': [],
    'joint_velocity': [],
    'gripper_states': [],
    'delta_position': [],
    'delta_end_effector_position': [],
    'delta_end_effector_orientation': []
}


# cube_pos, cube_ori = robot.getCubeStates()
# target_task_pos = cube_pos
# target_task_pos[2] += .5
action_taken = None
gripper_open = False
timestep = 0
while True:
    keys = p.getKeyboardEvents()
    delta_position = [0, 0, 0]
    if keys:
        timestep += 1
        loop_break = False
        for key, action in key_map.items():
            if key in keys and keys[key] & p.KEY_WAS_TRIGGERED:
                
                action_taken = action
                if action == 'open_gripper':
                    gripper_open = True
                    robot.control_gripper(gripper=True)
                    delta_position = [0, 0, 0]
                elif action == 'close_gripper':
                    gripper_open = False
                    robot.control_gripper(gripper=False)
                    delta_position = [0, 0, 0]
                elif action == 'stop':
                    loop_break = True
                    delta_position = [0, 0, 0]
                else:
                    ee_pos[0] += action[0]
                    ee_pos[1] += action[1]
                    ee_pos[2] += action[2]
                    delta_position = action
            
        print(ee_pos)
        # target_joint_positions = robot.InverseKinematics(ee_pos, [1, 0, 0, 0])
        # robot.setTargetPositions(target_joint_positions)
        orientation = [1, 0, 0, 0]
        robot.setEndEffectorPose(ee_pos, orientation)
        
        end_effector, _ = robot.getEndEffectorPose()
        delta_end_effector = np.array(end_effector) - np.array(previous_end_effector)
        previous_end_effector = end_effector
        for i in range(30):
            robot.step()

        # demo_data['images'].append(rgb_bn)
        # demo_data['depths'].append(depth_bn)
        print("timestep: ", timestep)
        demo_data['timestamps'].append(timestep)
        demo_data['joint_position'].append(robot.getJointStates()[0])
        demo_data['joint_velocity'].append(robot.getJointStates()[1])
        demo_data['delta_end_effector_position'].append(delta_position)
        demo_data['delta_end_effector_orientation'].append([1, 0, 0, 0])
        demo_data['delta_effector_positions'].append(delta_end_effector)
        demo_data['gripper_states'].append(gripper_open)
        demo_data['delta_position'].append(delta_position)

        
        time.sleep(stepsize)
        if loop_break:
            break
    
with open('demo/robot_demo_delta_position.pkl', 'wb') as f:
    pickle.dump(demo_data, f)

print("Done")