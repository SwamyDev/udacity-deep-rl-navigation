import numpy as np

from p1_navigation.memory import Memory
from tests.test_models import QModel


def _with_default(cfg, key, default):
    if key not in cfg:
        cfg[key] = default
    return cfg


def _with_mem_defaults(cfg):
    cfg = _with_default(cfg, 'batch_size', 64)
    cfg = _with_default(cfg, 'record_size', int(1e5))
    return cfg


def _with_model_default(cfg):
    return cfg


class DQNAgent:
    def __init__(self, observation_space, action_space, **kwargs):
        self._action_space = action_space
        self._memory = Memory(**_with_mem_defaults(kwargs))
        self._target_model = QModel(observation_space, action_space, **_with_model_default(kwargs.get('model', dict())))
        self._local_model = QModel(observation_space, action_space, **_with_model_default(kwargs.get('model', dict())))
        self._gamma = kwargs.get('gamma', 0.99)
        self._tau = kwargs.get('tau', 0.99)
        self._steps = 0

    def act(self, observation):
        return self._action_space.sample()

    def step(self, obs, action, reward, next_obs, done):
        self._memory.record(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

    def train(self):
        if self._memory.is_unfilled():
            return

        obs, actions, rewards, next_obs, dones = self._memory.sample()
        q_next = self._target_model.estimate(next_obs)
        q_target = rewards + (self._gamma * np.max(q_next, axis=1) * (1 - dones))
        self._local_model.fit(obs, actions, q_target)
        self._target_model.linear_interpolate(self._local_model, self._tau)
