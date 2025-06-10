import numpy as np
import transforms3d
import trimesh
import torch
from dinobot.env.base import SimpleEnv
# from modules.vision import DinoV2Encoder
from scipy.spatial.transform import Rotation as R
import sys
import os
import pybullet as p
# Redirect stdout and stderr to /dev/null
# sys.stdout = open(os.devnull, 'w')
# sys.stderr = open(os.devnull, 'w')

class CubeEnv(SimpleEnv):
    def __init__(self, render=False, Test_env=False):
        super().__init__(render, Test_env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Image based policy
        # self.encoder = DinoV2Encoder()

    def get_observations(self, state_type="state", render=False):

        if state_type == "image":
            image, depth = super().get_observations()
            if render:
                import matplotlib.pyplot as plt
                plt.imshow(image)
                plt.show()
            features = self.encoder.get_feature(image=image)
            # print("features: ", features.shape)
            img_features = features[(..., *([np.newaxis] * self.num_envs))]
            # print("img_features: ", img_features.shape)
            img_features = img_features.squeeze(-1)
            extras = {
                "observations": {
                    "critic": img_features,
                    "rnd_state": img_features
                }
            }
            return img_features, extras
        elif state_type == "state":
            joint_state, obj_pos, obj_ori = self.get_state()
            joint_state = np.array(joint_state)
            joint_state = joint_state.reshape(-1)
            obj_pos = np.array(obj_pos)
            obj_ori = np.array(obj_ori)
            # obj_state = np.concatenate([obj_pos, obj_ori])
            # print("joint_state: ", )
            # print("obj_pos: ", obj_pos)
            # print(joint_state.shape)
            joint_state = torch.tensor(joint_state, dtype=torch.float32).to(self.device)
            obj_state = torch.tensor(obj_pos, dtype=torch.float32).to(self.device)

            #Policy State
            # state = torch.cat([joint_state, obj_state], dim=0)
            state = joint_state
            state = state[(tuple([np.newaxis] * self.num_envs) +  (...,))]


            # RND State
            ee_pos, ee_ori = self.getEndEffectorPose()
            ee_pos = np.array(ee_pos)
            distance = np.linalg.norm(ee_pos - obj_pos)
            rnd_distance = min(distance, 0.3)
            rnd_distance_tensor = torch.tensor([rnd_distance], dtype=torch.float32).to(self.device)
            rnd_state = torch.cat([obj_state, rnd_distance_tensor], dim=0)
            rnd_state = rnd_state[(tuple([np.newaxis] * self.num_envs) +  (...,))]

            ee_pos = torch.tensor(ee_pos, dtype=torch.float32).to(self.device)
            MI_state = torch.cat([ee_pos, obj_state], dim=0)
            # print("MI_state: ", MI_state.shape)
            # rnd_state = torch.tensor(rnd_distance, dtype=torch.float32)
            # print("rnd_state: ", type(rnd_state), rnd_state)
            extras = {
                "observations": {
                    "critic": state,
                    "rnd_state": rnd_state,
                },
                "object_state": MI_state,
            }
            return state, extras
    
    def quaternion_distance(self, q1, q2):
        """Computes the geodesic distance between two quaternions."""
        q1 = R.from_quat(q1)
        q2 = R.from_quat(q2)
        relative_rotation = q1.inv() * q2
        angle = np.linalg.norm(relative_rotation.as_rotvec())  # Rotation vector norm gives angle
        return angle  # In radians

    def get_reward(self):
        done = False
        threshold = 0.05
        joint_state, obj_pos, obj_ori = self.get_state()
        obj_init_pos = np.array(self.object_initial_position)
        obj_init_ori = np.array(self.object_initial_orientation)
        obj_pos = np.array(obj_pos)
        obj_ori = np.array(obj_ori)
        pos_diff = np.linalg.norm(obj_pos - obj_init_pos)  # Euclidean distance
        ori_diff = self.quaternion_distance(obj_ori, obj_init_ori)  # Quaternion distance
        reward = pos_diff + ori_diff
        reward = np.clip(reward, 0, 1)
        # print("reward: ", reward)
        reward = 0
        # If the object is in contact with the robot, give a reward
        # if self.check_contact():
            # reward += 1
        # print("check_contact: ", self.check_contact())

        # If the object is moved to the right of the threshold, give a reward
        # if obj_pos[0] > threshold:
            # done = True
            # print("done", reward)
        return reward, done

    def step(self, action):
        # print("action", action)
        # action = action[0]
        position = action[:3]
        orientation = action[3:-1]
        gripper = action[-1]
        # print("gripper", gripper)
        gripper = 1 if gripper > 0.5 else 0
        # print("position", position)
        # print("orientation", orientation)
        # target_joint_positions = self.InverseKinematics(position, orientation)
        self.setDeltaEndControl(position, orientation)
        self.controlGripper(gripper)
        self.simulate_step()
        # for i in range(10):
        #     self.simulate_step()    
        reward, done = self.get_reward()
        reward = 0
        reward = np.array([reward])
        done = np.array([done])
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)
        # print("observation", self.get_observations().type())
        image_feature, extra = self.get_observations()
        return image_feature, reward, done, extra

    def get_move_toward_action(self):
        joint_state, obj_pos, obj_ori = self.get_state()
        joint_state = np.array(joint_state)
        obj_pos = np.array(obj_pos)
        ee_pos, ee_ori = self.getEndEffectorPose()
        ee_pos = np.array(ee_pos)

        direction = obj_pos - ee_pos
        direction = direction / np.linalg.norm(direction)

        delta_position = direction * 0.008
        delta_quaternion = np.array([0, 0, 0, 1])
        action = np.concatenate([delta_position, delta_quaternion, [0]])
        return action
    
    def check_contact(self):
        robot = self.robot
        object = self.object_id
        contacts = p.getContactPoints(robot, object)
        return len(contacts) > 0

    def get_distance(self):
        joint_state, obj_pos, obj_ori = self.get_state()
        joint_state = np.array(joint_state)
        obj_pos = np.array(obj_pos)
        ee_pos, ee_ori = self.getEndEffectorPose()
        ee_pos = np.array(ee_pos)
        distance = np.linalg.norm(ee_pos - obj_pos)
        return distance
    
    def replay_demo(self):
        """
        Inputs: demo: list of velocities that can then be executed by the end-effector.
        Replays a demonstration by moving the end-effector given recorded velocities.
        """
        import pickle
        with open('/home/gentlebear/Mres/dinobot/data_collection/demo/expert_demo_state.pkl', 'rb') as f:
            demo = pickle.load(f)
        # Get the recorded data from the pickle file
        new_action = np.array([0, 0, 0.01, 0, 0, 0, 1, 0])
        for i in range(20):
            demo["action"] = np.append(demo['action'], [new_action], axis=0)
        print("demo: ", demo["action"])
        episode_length = len(demo["action"])
        for step in range(1, episode_length):
            action = demo["action"][step-1]
            # print("base_action", base_action)
            next_state, _, done, extra = env.step(action)
        
        with open('/home/gentlebear/Mres/dinobot/data_collection/demo/expert_demo_state.pkl', 'wb') as f:
            pickle.dump(demo, f)

    
    def get_demo_action_for_state(self, state):
        """Find the demonstration action for the given state using nearest neighbor search."""
        # Convert state to numpy array if it's not already
        import pickle
        with open('/home/gentlebear/Mres/dinobot/data_collection/demo/expert_demo_state.pkl', 'rb') as f:
            demo = pickle.load(f)
        state = np.array(state)
        # print("state: ", state.shape)
        # Find the closest state from the demonstrations using Euclidean distance
        distances = np.linalg.norm(demo["state"] - state, axis=1)
        # print(demo["state"][0])
        closest_idx = np.argmin(distances)
        ee_pos, ee_ori = self.getEndEffectorPose()
        ee_pos = np.array(ee_pos)
        joint_state, obj_pos, obj_ori = self.get_state()
        obj_pos = np.array(obj_pos)
        distance = np.linalg.norm(ee_pos - obj_pos)
        print("distance: ", distance)
        # Get the demonstration action for the closest state
        # print(demo["action"][1])
        demo_action = demo["action"][closest_idx]
        print("closest_idx ", closest_idx)
        print("distances ", distances[closest_idx])
        # If the distance is large, we can return None (indicating no good match was found)
        if distances[closest_idx] > 0.05:  # You can tune the threshold for a good match
            return torch.tensor(np.zeros(8), dtype=torch.float32)
        
        return demo_action
    
if __name__ == "__main__":        
    env = CubeEnv(render=True, Test_env=True)
    # env.reset()
    action = env.replay_demo()
    # state = np.random.uniform(-1, 1, size=(1, 384))
    # env.replay_demo()
    # for i in range(100):
    #     action = env.move_towards()
    #     print("action: ", action)
    #     _, reward, done, info = env.step(action)
    # for i in range(100):
    #     state, _  = env.get_observations()
    #     # print("state: ", state)
    #     # print(type(state))
    #     state = state.cpu().detach().numpy()
    #     action = env.get_demo_action_for_state(state)
    #     print("action: ", action)
    #     _, reward, done, info = env.step(action)
    #     print("reward: ", reward)
    print("done")