import numpy as np
from matplotlib import pyplot as plt

# Load the saved trajectory data
restored_actions_per_skill = np.load('trajectory_data.npy', allow_pickle=True)

# Example of accessing restored trajectory for a specific skill (e.g., Skill 0)
skill_0_trajectory = restored_actions_per_skill[0]

# Now you can use this trajectory for further evaluation or plotting
print(skill_0_trajectory)
