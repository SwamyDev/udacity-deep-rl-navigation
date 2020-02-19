import abc
import collections
import numpy as np
from abc import ABC

from udacity_rl.memory import Memory


def with_default(cfg, key, default):
    if key not in cfg:
        cfg[key] = default
    return cfg


class Agent(abc.ABC):
    def __init__(self, observation_size, action_size, **kwargs):
        self._ctr_config = kwargs
        self._observation_size = observation_size
        self._action_size = action_size

    @property
    def configuration(self):
        return {'observation_size': self._observation_size, 'action_size': self._action_size, **self._ctr_config}

    @abc.abstractmethod
    def act(self, observation, epsilon=0):
        pass

    @abc.abstractmethod
    def step(self, obs, action, reward, next_obs, done):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def save(self, save_path):
        pass

    @abc.abstractmethod
    def load(self, save_path):
        pass


def _with_mem_defaults(cfg):
    cfg = with_default(cfg, 'batch_size', 64)
    cfg = with_default(cfg, 'record_size', int(1e5))
    return cfg


def _only_memory_args(mem_cfg):
    return {k: mem_cfg[k] for k in mem_cfg if k in Memory.__init__.__code__.co_varnames}


class MemoryAgent(Agent, ABC):
    def __init__(self, observation_size, action_size, **kwargs):
        super().__init__(observation_size, action_size, **kwargs)
        mem_cfg = _with_mem_defaults(kwargs)
        self._memory = Memory(**_only_memory_args(mem_cfg))

    def step(self, obs, action, reward, next_obs, done):
        if not isinstance(action, (collections.Sequence, np.ndarray)):
            action = [action]

        self._memory.record(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
