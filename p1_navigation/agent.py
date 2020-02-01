import collections

import numpy as np

from p1_navigation.memory import Memory
from p1_navigation.model import QModel


def _with_default(cfg, key, default):
    if key not in cfg:
        cfg[key] = default
    return cfg


def _with_mem_defaults(cfg):
    cfg = _with_default(cfg, 'batch_size', 64)
    cfg = _with_default(cfg, 'record_size', int(1e5))
    return cfg


def _with_model_default(cfg):
    cfg = _with_default(cfg, 'layers', [{'activation': 'relu', 'size': 64}])
    return cfg


class DQNAgent:
    def __init__(self, observation_space, action_space, **kwargs):
        self._action_space = action_space
        self._memory = Memory(**_with_mem_defaults(kwargs))
        obs_size = observation_space.n if hasattr(observation_space, 'n') else observation_space.shape[0]
        self._target_model = QModel(obs_size, action_space.n, **_with_model_default(kwargs.get('model', dict())))
        self._local_model = QModel(obs_size, action_space.n, **_with_model_default(kwargs.get('model', dict())))
        self._gamma = kwargs.get('gamma', 0.99)
        self._tau = kwargs.get('tau', 0.99)
        self._steps = 0

    def act(self, observation):
        q_values = self._target_model.estimate([observation]).squeeze()
        return np.argmax(q_values)

    def step(self, obs, action, reward, next_obs, done):
        if not isinstance(action, (collections.Sequence, np.ndarray)):
            action = [action]

        self._memory.record(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        self._steps = 1

    def train(self):
        if self._memory.is_unfilled():
            return

        obs, actions, rewards, next_obs, dones = self._memory.sample()
        q_next = self._target_model.estimate(next_obs)
        q_target = rewards + (self._gamma * np.max(q_next, axis=1) * (1 - dones))
        self._local_model.fit(obs, actions, q_target)
        self._target_model.linear_interpolate(self._local_model, self._tau)
