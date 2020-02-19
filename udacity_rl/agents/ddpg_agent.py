import numpy as np

from udacity_rl.agents.agent import MemoryAgent


class DDPGAgent(MemoryAgent):
    def __init__(self, observation_size, action_space, **kwargs):
        super().__init__(observation_size, action_space.shape[0], **kwargs)
        self._action_space = action_space

    def act(self, observation, epsilon=0):
        return np.random.random(self._action_size) * 2 - np.ones(self._action_size)

    def train(self):
        pass

    def save(self, save_path):
        pass

    def load(self, save_path):
        pass
