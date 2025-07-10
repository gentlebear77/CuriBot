import random
from collections import namedtuple
import numpy as np
Transition = namedtuple('Transition', ('state', 'z', 'done', 'action', 'object_state', 'next_state'))


class Memory:
    def __init__(self, buffer_size, seed):
        self.buffer_size = buffer_size
        self.buffer = []
        self.seed = seed
        random.seed(self.seed)

    def add(self, *transition):
        self.buffer.append(Transition(*transition))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        assert len(self.buffer) <= self.buffer_size

    
    def sample(self, size):
        # return random.sample(self.buffer, size)
        # Group transitions by episode_id
        # episodes = {}
        # for transition in self.buffer:
        #     episodes.setdefault(transition.episode_id, []).append(transition)
        # episodes = np.array(list(episodes.values()))
        # print("episodes", episodes.shape)
        # print("episodes", episodes[0])
        # Choose a random episode
        current_episode_ids = len(self.buffer)
        # print("current_episode_ids", current_episode_ids)
        random_episode_ids = np.random.randint(0, current_episode_ids, size)
        # print("random_episode_ids", random_episode_ids) 
        # Collect transitions from the sampled episode IDs
        
        sampled_transitions = [self.buffer[i] for i in random_episode_ids]


        # for episode_id in random_episode_ids:
        #     episode_transitions = self.buffer[episode_id]
        #     # print("episode_transitions", episode_transitions)
        #     sampled_transitions.extend(episode_transitions)
        # print("sampled_transitions", len(sampled_transitions))
        # Sample from that episode
        # print("size, episode_transitions", size, len(episode_transitions))
        return sampled_transitions

    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer = []

    @staticmethod
    def get_rng_state():
        return random.getstate()

    @staticmethod
    def set_rng_state(random_rng_state):
        random.setstate(random_rng_state)
