import numpy as np
from udacity_rl.agents.agent import Agent


class DDPGAgent(Agent):
    def __init__(self, observation_size, action_size, **kwargs):
        super().__init__(observation_size, action_size, **kwargs)

    def act(self, observation, epsilon=0):
        return np.random.random(self._action_size) * 2 - np.ones(self._action_size)

    def step(self, obs, action, reward, next_obs, done):
        pass

    def train(self):
        pass

    def save(self, save_path):
        pass

    def load(self, save_path):
        pass