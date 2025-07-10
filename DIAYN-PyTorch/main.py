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
    # if distances[closest_idx] > 0.1:  # You can tune the threshold for a good match
    #     # if self.env.check_contact():
    #     #     # print("distance", self.env.get_distance())
    #     #     return torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device={"cuda" if torch.cuda.is_available() else "cpu"})
    #     move_toward_action = env.get_move_toward_action()
    #     move_toward_action = torch.tensor([move_toward_action], dtype=torch.float32, device="cpu")
    #     return move_toward_action
    demo_action = torch.tensor([demo_action], dtype=torch.float32, device="cpu")
    return demo_action

class Logger:
    def __init__(self, agent, **params):
        self.agent = agent
        self.params = params

    def on(self):
        pass

    def log(self, episode, reward, z, logq_z, logmi, step, np_rng_state, *rng_states):
        wandb.log({
            'episode': episode,
            'reward': reward,
            'z': z,
            'logq_z': logq_z,
            'logmi': logmi,
            'step': step,
        })
        print(f"Episode {episode}, logq_z: {logq_z}, logmi: {logmi}")

    def load_weights(self):
        # Mock loading of previous weights
        return 0, None, None, None, None, None, None
    

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
        "MI_scale": 1000,
        "alpha": 0.1,
        "gamma": 0.99,
        "tau": 0.005,
        "max_n_episodes": 300,
        "action_bounds": [-0.01, 0.01],
        "epoch_length": 1,
        "train_interval": 1,
    }


wandb.init(project="DIYAN")
wandb.config.update(get_params())
import os
import datetime
def save_policy(agent, interval, episode, save_dir='./checkpoints'):
    # Create the save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    if episode % interval == 0:
        # Save the policy
        policy_path = os.path.join(save_dir, f"joint_policy_{episode}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pt")
        torch.save(agent.policy_network.state_dict(), policy_path)
        MI_policy_path = os.path.join(save_dir, f"MI_policy_{episode}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pt")
        torch.save(agent.MI_discriminator.state_dict(), MI_policy_path)
        print(f"Policy saved at {policy_path}")
        
        # Log the policy to WandB
        wandb.save(policy_path)

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    params = get_params()

    env = CubeEnv(render=False, Test_env=True)
    obs, extras = env.get_observations()
    n_states = obs.shape[1]
    n_object_states = extras["object_state"].shape[0]
    # print("n_object_states", n_object_states)
    n_actions = env.num_actions
    action_bounds = [-0.01, 0.01]

    import pickle
    with open('/home/gentlebear/Mres/dinobot/data_collection/demo/expert_demo_state.pkl', 'rb') as f:
        demo_data = pickle.load(f)
    
    episode_length = demo_data["action"].shape[0]
    print("Demo data loaded with", episode_length, "length.")
    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "n_object_states": n_object_states,
                   "action_bounds": action_bounds,
                   "episode_length": episode_length,})
    print("params:", params)
    del n_states, n_actions, action_bounds

    all_episodes_data = []

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, **params)
    logger = Logger(agent, **params)

    if params["do_train"]:

        if not params["train_from_scratch"]:
            
            with open('/home/gentlebear/Mres/DIAYN-PyTorch/checkpoints/MI_policy_100_2025-06-26-02-04-19.pt', 'rb') as f:
                agent.MI_discriminator.load_state_dict(torch.load(f))
            min_episode = 0
            last_logq_zs = 0
            last_mi_loss = 0
            np.random.seed(params["seed"])
            print("Keep training from previous MI estimator.")

        else:
            min_episode = 0
            last_logq_zs = 0
            last_mi_loss = 0
            np.random.seed(params["seed"])
            # env.seed(params["seed"])
            # env.observation_space.seed(params["seed"])
            # env.action_space.seed(params["seed"])
            print("Training from scratch.")

        logger.on()
 
        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            
            # if episode % 100 == 0:
                # agent.clear_memory()
            episode_rollout = {
                "states": [],
                "actions": [],
                "next_states": [],
                "object_states": [],
                "episode_ids": [],
                "z": [],
                "done": []}

            z = np.random.choice(params["n_skills"], p=p_z)
            env.reset()
            state, _ = env.get_observations()
            state = state.cpu().detach().numpy()
            raw_state = state[0]
            state = concat_state_latent(raw_state, z, params["n_skills"])
            episode_reward = 0
            logq_zses = []
            mi_losses = []
            max_n_steps = 150
            for step in range(1, 1 + episode_length):
    
                residual_action = agent.choose_action(state)
                # residual_action[2] = 0.0  # Set the z-axis action to 0.0
                base_action = demo_data["action"][step-1]
                # distance = env.get_distance()
                # print("distance", distance)
                action = base_action + residual_action
                # action = base_action
                next_state, _, done, extra = env.step(action)
                if step == episode_length + 1:
                    done = True
                object_state = extra["object_state"]
                # print("object_state", object_state[3:6])
                # print("effector_state", object_state[:3])
                next_state = next_state.cpu().detach().numpy()
                object_state = object_state.cpu().detach().numpy()
                # print("state", state)
                # print("object_state", object_state)
                # print("next_state", next_state)
                # print("action", action)
                raw_state = next_state[0]
                # print("raw_state, next_state", raw_state, next_state)
                next_state = concat_state_latent(raw_state, z, params["n_skills"])
                # print("next_state", next_state.shape)
                # print("state", state.shape)
                # agent.store(state, z, done, residual_action, object_state, next_state, episode)
                # mi_value = agent.run_MI_discriminator(object_state[:3], object_state[3:6])
                # print("mi_value", mi_value)
                # store the rollout data
                # print("object_state", object_state)
                episode_rollout["states"].append(state)
                episode_rollout["actions"].append(residual_action)
                episode_rollout["next_states"].append(next_state)
                episode_rollout["object_states"].append(object_state)
                episode_rollout["z"].append(z)
                episode_rollout["done"].append(done)

                state = next_state.copy()
 
            agent.store(episode_rollout["states"], episode_rollout["z"], episode_rollout["done"],
                        episode_rollout["actions"], episode_rollout["object_states"], episode_rollout["next_states"])
            
            if episode % params["train_interval"] == 0:
                for i in range (params["epoch_length"]):
                    loss = agent.train()
                    if loss is None:
                        logq_zses.append(last_logq_zs)
                        mi_losses.append(last_mi_loss)
                    else:
                        logq_zs, mi_loss = loss
                        logq_zses.append(logq_zs)
                        mi_losses.append(mi_loss)
                save_policy(agent, params["interval"], episode)
                logger.log(episode,
                        episode_reward,
                        z,
                        sum(logq_zses) / len(logq_zses),
                        sum(mi_losses) / len(mi_losses),
                        step,
                        np.random.get_state(),
                        *agent.get_rng_states(),
                        )
                logq_zses = []
                mi_losses = []

    wandb.finish()
