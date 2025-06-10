import pickle
import numpy as np

with open('/home/gentlebear/Mres/dinobot/data_collection/demo/expert_demo_state.pkl', 'rb') as f:
    demo = pickle.load(f)

episode_length = demo["state"].shape[0]
for i in range(episode_length):
    new_action = demo["action"][i]
    # new_action[3] = 0.0
    # new_action[-2] = 1.0
    # print(new_action)
    # demo["action"][i] = new_action
    print(demo["action"][i])

# with open('/home/gentlebear/Mres/dinobot/data_collection/demo/expert_demo_state.pkl', 'wb') as f:
#     pickle.dump(demo, f)