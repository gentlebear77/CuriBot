import torch
import os
import yaml
import argparse
from dinobot.env.base import SimpleEnv
from dinobot.env.cube import CubeEnv
from modules import ActorCritic, ActorCriticRecurrent
from runners import OnPolicyRunner  # Ensure this class is saved in a file named `on_policy_runner.py`

# Load training configuration
def load_config(config_path="config/dummy_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(args):
     # Load configuration
    config = load_config("config/dummy_config.yaml")

    # Set device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create environment
    # env = VecEnv(**config["environment"])
    video = args.video
    env = CubeEnv(render=True)
    # Set up log directory
    log_dir = config.get("log_dir", "checkpoints/")
    os.makedirs(log_dir, exist_ok=True)

    # Initialize runner
    runner = OnPolicyRunner(env, config, log_dir=log_dir, device=device, video=video)

    # Start training
    num_iterations = config.get("num_iterations", 1000)
    runner.learn(num_iterations)
    import time
    path = f'checkpoints/PPO_{int(time.time())}.pth'
    runner.save(path=path)
    env.disconnect()
    # runner.load(path = path)
    # Evaluate the trained model
    runner.evaluate(env = CubeEnv(render=True), num_episodes = 1, render = True)


parser = argparse.ArgumentParser(description="Training script with rendering option.")
parser.add_argument("--video", type=lambda x: (str(x).lower() == 'true'), default=False, help="Enable rendering (True/False)")
args = parser.parse_args()

if __name__ == "__main__":
   main(args)