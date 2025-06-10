import torch
import os
import yaml
import gym
from dinobot.env.base import SimpleEnv
from dinobot.env.cube import CubeEnv
from modules import ActorCritic, ActorCriticRecurrent
from runners import OnPolicyRunner  # Ensure this class is saved in a file named `on_policy_runner.py`

# Load training configuration
def load_config(config_path="config/dummy_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # Load configuration
    config = load_config("config/dummy_config.yaml")

    # Set device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create environment
    # env = VecEnv(**config["environment"])
    env = CubeEnv()
    # Set up log directory
    log_dir = config.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    # Initialize runner
    runner = OnPolicyRunner(env, config, log_dir=log_dir, device=device)

    # Start training
    # num_iterations = config.get("num_iterations", 5)
    # runner.learn(num_iterations)
    import time
    # runner.save(path=f'checkpoints/PPO_{int(time.time())}.pth')
    env.disconnect()
    runner.load(path = "/home/gentlebear/Mres/dinobot/checkpoints/PPO_1741201029.pth")
    # Evaluate the trained model
    runner.evaluate(env = CubeEnv(render=True), num_episodes = 1, render = True)
