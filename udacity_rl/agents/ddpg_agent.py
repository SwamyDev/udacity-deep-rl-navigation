import logging
import numpy as np

from udacity_rl.agents.agent import MemoryAgent
from udacity_rl.agents.algorithms.ddpg import DDPGAlgorithm
from udacity_rl.model import Actor, Critic

logger = logging.getLogger(__name__)

DEFAULT_ACTOR_CFG = {
    "layers": [
        {
            "activation": "relu",
            "size": 64
        },
        {
            "activation": "relu",
            "size": 64
        }
    ],
    "device": "cpu",
    "lr": 1e-4
}

DEFAULT_CRITIC_CFG = {
    "layers": [
        {
            "activation": "leaky_relu",
            "size": 64
        },
        {
            "activation": "leaky_relu",
            "size": 64
        }
    ],
    "device": "cpu",
    "lr": 1e-3
}


class DDPGAgent(MemoryAgent):
    def __init__(self, observation_space, action_space, actor=None, critic=None, **kwargs):
        super().__init__(observation_space, action_space, actor=actor, critic=critic, **kwargs)

        actor = actor or DEFAULT_ACTOR_CFG
        critic = critic or DEFAULT_CRITIC_CFG
        self._algorithm = DDPGAlgorithm(Actor(self._observation_size, self._action_size, **actor),
                                        Actor(self._observation_size, self._action_size, **actor),
                                        Critic(self._observation_size, self._action_size, **critic),
                                        Critic(self._observation_size, self._action_size, **critic),
                                        kwargs.get('gamma', 0.99),
                                        kwargs.get('tau', 1e-3))

    def _print_config(self):  # pragma: no cover
        logger.info(f"DDPG configuration:\n"
                    f"\tObservation Size:\t{self._observation_size}\n"
                    f"\tAction Size:\t\t{self._action_size}\n")

    def act(self, observation, epsilon=0):
        action = self._algorithm.estimate(observation)
        if epsilon:
            action += epsilon * self.action_space.sample()
        return np.clip(action, self.action_space.low, self.action_space.high)

    def train(self):
        if self._memory.is_unfilled():
            return

        self._algorithm.fit(*self._memory.sample())

    def save(self, save_path):
        self._algorithm.save_models(save_path)

    def load(self, save_path):
        self._algorithm.load_models(save_path)
