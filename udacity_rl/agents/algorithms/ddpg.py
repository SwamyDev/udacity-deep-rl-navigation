import logging
import numpy as np

import torch

logger = logging.getLogger(__name__)


class DDPGAlgorithm:
    def __init__(self, actor_local, actor_target, critic_local, critic_target, gamma, tau):
        self._actor_local = actor_local
        self._actor_target = actor_target
        self._critic_local = critic_local
        self._critic_target = critic_target
        self._gamma = gamma
        self._tau = tau
        self._print_config()

    def _print_config(self):
        logger.info(f"DDPGAlgorithm configuration:\n"
                    f"\tGamma:\t{self._gamma}\n"
                    f"\tTau:\t\t{self._tau}\n")

    def estimate(self, observation):
        return self._actor_local.estimate(observation)

    def fit(self, obs, actions, rewards, next_obs, dones):
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

    def save_models(self, save_path):
        self._actor_local.save(save_path / 'actor_local.pth')
        self._actor_target.save(save_path / 'actor_target.pth')
        self._critic_local.save(save_path / 'critic_local.pth')
        self._critic_target.save(save_path / 'critic_target.pth')

    def load_models(self, save_path):
        self._actor_local.load(save_path / 'actor_local.pth')
        self._actor_target.load(save_path / 'actor_target.pth')
        self._critic_local.load(save_path / 'critic_local.pth')
        self._critic_target.load(save_path / 'critic_target.pth')
