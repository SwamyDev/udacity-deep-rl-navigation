import logging
import numpy as np
import torch

from udacity_rl.agents.agent import MemoryAgent
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
        self._actor_local = Actor(self._observation_size, self._action_size, **actor)
        self._actor_target = Actor(self._observation_size, self._action_size, **actor)

        critic = critic or DEFAULT_CRITIC_CFG
        self._critic_local = Critic(self._observation_size, self._action_size, **critic)
        self._critic_target = Critic(self._observation_size, self._action_size, **critic)

        self._gamma = kwargs.get('gamma', 0.99)
        self._tau = kwargs.get('tau', 1e-3)
        self._print_config()

    def _print_config(self):
        logger.info(f"DDPG configuration:\n"
                    f"\tObservation Size:\t{self._observation_size}\n"
                    f"\tAction Size:\t\t{self._action_size}\n"
                    f"\tGamma:\t\t\t{self._gamma}\n"
                    f"\tTau:\t\t\t{self._tau}\n")

    def act(self, observation, epsilon=0):
        action = self._actor_local.estimate(observation)
        if epsilon:
            action += epsilon * self.action_space.sample()
        return np.clip(action, self.action_space.low, self.action_space.high)

    def train(self):
        if self._memory.is_unfilled():
            return

        obs, actions, rewards, next_obs, dones = self._memory.sample()
        obs = torch.from_numpy(np.vstack(obs)).float().to(self._actor_target.device)
        actions = torch.from_numpy(np.vstack(actions)).float().to(self._actor_target.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self._actor_target.device)
        next_obs = torch.from_numpy(np.vstack(next_obs)).float().to(self._actor_target.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self._actor_target.device)

        next_acts = self._actor_target(next_obs)
        next_q_vals = self._critic_target(next_obs, next_acts)
        q_targets = rewards + self._gamma * (1 - dones) * next_q_vals
        self._critic_local.fit(obs, actions, q_targets)

        estimated_acts = self._actor_local(obs)
        actor_loss = -self._critic_local(obs, estimated_acts).mean()
        self._actor_local.minimize(actor_loss)

        self._critic_target.linear_interpolate(self._critic_local, self._tau)
        self._actor_target.linear_interpolate(self._actor_local, self._tau)

    def save(self, save_path):
        self._actor_local.save(save_path / 'actor_local.pth')
        self._actor_target.save(save_path / 'actor_target.pth')
        self._critic_local.save(save_path / 'critic_local.pth')
        self._critic_target.save(save_path / 'critic_target.pth')

    def load(self, save_path):
        self._actor_local.load(save_path / 'actor_local.pth')
        self._actor_target.load(save_path / 'actor_target.pth')
        self._critic_local.load(save_path / 'critic_local.pth')
        self._critic_target.load(save_path / 'critic_target.pth')
