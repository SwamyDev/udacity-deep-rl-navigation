import logging
import random
import numpy as np

from udacity_rl.agents.agent import with_default, MemoryAgent
from udacity_rl.model import QModel

logger = logging.getLogger(__name__)


def _with_model_default(cfg):
    cfg = with_default(cfg, 'layers', [{'activation': 'relu', 'size': 64}])
    return cfg


class DQNAgent(MemoryAgent):
    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self._target_model = QModel(self._observation_size, self._action_size,
                                    **_with_model_default(kwargs.get('model', dict())))
        self._local_model = QModel(self._observation_size, self._action_size,
                                   **_with_model_default(kwargs.get('model', dict())))
        self._gamma = kwargs.get('gamma', 0.99)
        self._tau = kwargs.get('tau', 0.1)

        self._print_config()

    def _print_config(self):
        logger.info(f"DQNAgent configuration:\n"
                    f"\tObservation Size:\t{self._observation_size}\n"
                    f"\tAction Size:\t\t{self._action_size}\n"
                    f"\tGamma:\t\t\t{self._gamma}\n"
                    f"\tTau:\t\t\t{self._tau}\n")

    def act(self, observation, epsilon=0):
        if random.random() > epsilon:
            q_values = self._target_model.estimate([observation]).squeeze()
            return np.argmax(q_values)
        else:
            return np.random.randint(self._action_size)

    def train(self):
        if self._memory.is_unfilled():
            return

        obs, actions, rewards, next_obs, dones = self._memory.sample()
        q_next = self._target_model.estimate(next_obs)
        q_target = rewards + (self._gamma * np.max(q_next, axis=1) * (1 - dones))
        self._local_model.fit(obs, actions, q_target)
        self._target_model.linear_interpolate(self._local_model, self._tau)

    def save(self, save_path):
        self._target_model.save(save_path / 'target.pth')
        self._local_model.save(save_path / 'local.pth')

    def load(self, save_path):
        self._target_model.load(save_path / 'target.pth')
        self._local_model.load(save_path / 'local.pth')
