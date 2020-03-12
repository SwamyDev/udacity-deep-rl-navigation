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


class MADDPGAgent(MemoryAgent):
    def __init__(self, observation_space, action_space, actor=None, critic=None, **kwargs):
        super().__init__(observation_space, action_space, actor=actor, critic=critic, **kwargs)

        self._num_agents = self.observation_space.shape[0]
        self._observation_size = self.observation_space.shape[1]
        self._action_size = self.action_space.shape[1]

        actor = actor or DEFAULT_ACTOR_CFG
        critic = critic or DEFAULT_CRITIC_CFG

        self._algorithms = []
        for i in range(self._num_agents):
            self._algorithms.append(DDPGAlgorithm(Actor(self._observation_size, self._action_size, **actor),
                                                  Actor(self._observation_size, self._action_size, **actor),
                                                  Critic(self._observation_size, self._action_size, **critic),
                                                  Critic(self._observation_size, self._action_size, **critic),
                                                  kwargs.get('gamma', 0.99),
                                                  kwargs.get('tau', 1e-3)))

        self._preheat_steps = kwargs.get('preheat_steps', 10000)
        self._step = 0

    def _print_config(self):
        logger.info(f"MADDPG configuration:\n"
                    f"\tNumber of agents:\t{self._num_agents}\n"
                    f"\tObservation Size:\t{self._observation_size}\n"
                    f"\tAction Size:\t\t{self._action_size}\n")

    def act(self, observation, epsilon=0):
        if epsilon and self._step < self._preheat_steps:
            self._step += 1
            return self.action_space.sample()

        action = np.empty(shape=self.action_space.shape)
        for i in range(self._num_agents):
            action[i] = self._algorithms[i].estimate(observation[i])

        if epsilon:
            action += epsilon * np.random.randn(*self.action_space.shape)
        return np.clip(action, -1.0, 1.0)

    def train(self):
        if self._memory.is_unfilled():
            return

        obs, action, reward, next_obs, done = self._memory.sample()
        for i in range(self._num_agents):
            self._algorithms[i].fit(obs[:, i, :], action[:, i, :], reward[:, i], next_obs[:, i, :], done)
            
    def save(self, save_path):
        for i in range(self._num_agents):
            p = save_path / f"mind_{i}"
            p.mkdir(exist_ok=True)
            self._algorithms[i].save_models(save_path / f"mind_{i}")

    def load(self, save_path):
        for i in range(self._num_agents):
            self._algorithms[i].load_models(save_path / f"mind_{i}")
