# python3 main.py --mem_size=1000000 --interval=100 --do_train --n_skills=20 --train_from_scratch

from Brain import SACAgent
import numpy as np
from tqdm import tqdm
from dinobot.env.cube import CubeEnv
import torch
import wandb

def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])

def get_demo_action_for_state(self, state, demo):
    """Find the demonstration action for the given state using nearest neighbor search."""
    # Convert state to numpy array if it's not already
    # state = state.cpu().detach().numpy()
    # state = np.array(state)
    
    # Find the closest state from the demonstrations using Euclidean distance
    distances = np.linalg.norm(demo["state"] - state, axis=1)
    closest_idx = np.argmin(distances)
    
    # Get the demonstration action for the closest state
    demo_action = demo["action"][closest_idx]
    dis_threshold = 0.12
    # print("demo_action", demo_action.shape)
    # If the distance is large, we can return None (indicating no good match was found)
    if distances[closest_idx] > 0.1:  # You can tune the threshold for a good match
        # if self.env.check_contact():
        #     # print("distance", self.env.get_distance())
        #     return torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device={"cuda" if torch.cuda.is_available() else "cpu"})
        move_toward_action = env.get_move_toward_action()
        move_toward_action = torch.tensor([move_toward_action], dtype=torch.float32, device="cpu")
        return move_toward_action
    demo_action = torch.tensor([demo_action], dtype=torch.float32, device="cpu")
    return demo_action

def get_params():
    return {
        "n_skills": 5,
        "mem_size": 1000000,
        "interval": 100,
        "do_train": True,
        "train_from_scratch": True,
        "seed": 42,
        "batch_size": 256,
        "n_hiddens": 256,
        "lr": 0.001,
        "reward_scale": 0.01,
        "MI_scale": 1,
        "alpha": 0.1,
        "gamma": 0.99,
        "tau": 0.005,
        "max_n_episodes": 100,
        "action_bounds": [-0.01, 0.01],
        "epoch_length": 1,
        "train_interval": 1,
    }


if __name__ == "__main__":
    params = get_params()

    env = CubeEnv(render=True, Test_env=True)
    obs, extras = env.get_observations()
    n_states = obs.shape[1]
    n_object_states = extras["object_state"].shape[0]

    n_actions = env.num_actions
    action_bounds = [-0.005, 0.005]

    new_demo = {
        "state": [],
        "action": []
    }
    import pickle
    with open('/home/gentlebear/Mres/dinobot/data_collection/demo/expert_demo_state.pkl', 'rb') as f:
        demo_data = pickle.load(f)
    # print("demo_data", demo_data)
    episode_length = demo_data["action"].shape[0]

    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "n_object_states": n_object_states,
                   "action_bounds": action_bounds,
                   "episode_length": episode_length,})
    print("params:", params)
    del n_states, n_actions, action_bounds

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, **params)

    with open('/home/gentlebear/Mres/DIAYN-PyTorch/checkpoints/joint_policy_100_2025-07-10-02-20-07.pt', 'rb') as f:
        agent.policy_network.load_state_dict(torch.load(f))

    # with open('/home/gentlebear/Mres/DIAYN-PyTorch/checkpoints/MI_policy_100_2025-06-26-02-04-19.pt', 'rb') as f:
        # agent.MI_discriminator.load_state_dict(torch.load(f))
    

    actions_per_skill = []

    for z in range(params["n_skills"]):
        env.reset()
        object_state_rollout = []
        state, _ = env.get_observations()
        state = state.cpu().detach().numpy()
        raw_state = state[0]
        state = concat_state_latent(raw_state, z, params["n_skills"])
        episode_reward = 0
        logq_zses = []
        print(f"Skill {z}")
        max_n_steps = 150
        skill_actions = []
        coord = np.zeros(3)
        
        env.Videosave_start(video_name=f"joint_DIAYN{z}")
        # env.Videosave_start(video_name=f"joint_DIAYN{z}")
        for step in range(1, 1 + episode_length):
            residual_action = agent.choose_action(state)
            # print("residual_action", residual_action)
            # base_action = get_demo_action_for_state(agent, raw_state, demo_data)

            base_action = demo_data["action"][step-1]
            
            distance = env.get_distance()
            # residual_action[2] = 0.0  # Set the z-axis action to 0.0
            action = residual_action + base_action
            # action = base_action
            # print("distance", distance)
            # print("action", action)
            delta_action = action[:3]
            coord += delta_action
            # print("coord", coord)
            skill_actions.append(coord.copy())
            next_state, _, done, extra = env.step(action)
            object_state = extra["object_state"]
            # next_object_state = next_object_state.cpu().detach().numpy()
            object_state_rollout.append(object_state.cpu().detach().numpy())
            next_state = next_state.cpu().detach().numpy()
            raw_state = next_state[0]
            next_state = concat_state_latent(raw_state, z, params["n_skills"])

            T = env.get_transformation_base_to_ee()
            ee_pos, ee_ori, obj_pos, obj_ori = env.get_MI_state()
            state_base = (ee_pos, ee_ori)  # Example state in base frame
            obj_base = (obj_pos, obj_ori)  # Example object state in base frame
            # Transform the state to end-effector frame
            state_ee, state_ori = env.transform_state_to_ee_frame(state_base, T)
            state_ee_obj, state_ori_obj = env.transform_state_to_ee_frame(obj_base, T)
            print("Transformed Position in EE frame:", state_ee_obj)
            print("Transformed Position in EE frame:", state_ee)

            # print("object_state", object_state)
            # MI_value = agent.run_MI_discriminator(object_state[3:6], object_state[:3])
            # MI_reward = torch.clip(MI_value, min=0, max=1) - 1.0
            # print("MI_value", MI_value)
            # print("mi_value", MI_reward)
            # print("action", type(action))
            # print("state", type(state))
            # print("next_state", type(next_state))
            new_demo["state"].append(state.tolist())
            new_demo["action"].append(action.tolist())
            state = next_state
            if done:
                break
        env.Videosave_end()
        # print("new_demo", new_demo)
        # with open('/home/gentlebear/Mres/DIAYN-PyTorch/demo_buffer/learned_can_demo.pkl', 'wb') as f:
            # pickle.dump(new_demo, f)
        # exit(0)
        object_state_rollout = np.array(object_state_rollout)
        object_state_rollout = torch.tensor(object_state_rollout)
        object_state_rollout = object_state_rollout.unsqueeze(0)
        # print("object_state_rollout", object_state_rollout.shape)
        MI_values = []
        for i in range(object_state_rollout.shape[1] - 1):
            # print(object_state_rollout[:, i, :].shape, object_state_rollout[:, i + 1, :].shape)
            object_state = torch.cat((object_state_rollout[:, i, :], object_state_rollout[:, i + 1, :]), dim=0)
            # object_state = object_state_rollout[:, i, :]
            object_state = object_state.unsqueeze(0)
            # print("object_state", object_state)
            # print("object_state", object_state.shape)
            MI_value = agent.run_MI_discriminator(object_state[:, :, :3], object_state[:, :, 3:6])
            print(f'MI_value {i}:', MI_value)
            MI_values.append(MI_value.item())
        # Now plot the values
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        plt.figure(figsize=(8, 4))
        plt.plot(MI_values, marker='o', linestyle='-', color='blue')
        plt.title("Mutual Information Values Over Time")
        plt.xlabel("Time Step (i)")
        plt.ylabel("MI Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # MI_value = agent.run_MI_discriminator(object_state_rollout[:, :, 3:6], object_state_rollout[:, :, :3])
        # print("MI_value", MI_value)
        # print("object_state_rollout", object_state_rollout.shape)
        # env.Videosave_end()
        # print(skill_actions)
        actions_per_skill.append(np.array(skill_actions))
        

    np.save('trajectory_data_0.npy', actions_per_skill)
    # Visualize the trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for z, skill_actions in enumerate(actions_per_skill):
        skill_actions = np.array(skill_actions)
        
        ax.plot(skill_actions[:, 0], skill_actions[:, 1],skill_actions[:, 2], label=f"Skill {z}")

    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    ax.set_title('3D Trajectory of Skills')
    ax.legend()
    plt.show()