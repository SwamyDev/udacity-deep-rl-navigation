import collections
import json
import random

import numpy as np

from udacity_rl.memory import Memory
from udacity_rl.model import QModel


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


def _only_memory_args(mem_cfg):
    return {k: mem_cfg[k] for k in mem_cfg if k in Memory.__init__.__code__.co_varnames}


class DefaultEpsilonCalc:
    def __init__(self, start, end, decay):
        self._eps = start
        self._end = end
        self._decay = decay

    def __call__(self):
        self._eps = max(self._eps * self._decay, self._end)
        return self._eps


class DQNAgent:
    def __init__(self, observation_size, action_size, **kwargs):
        self._ctr_config = kwargs
        mem_cfg = _with_mem_defaults(kwargs)
        self._observation_size = observation_size
        self._action_size = action_size
        self._memory = Memory(**_only_memory_args(mem_cfg))
        self._target_model = QModel(observation_size, action_size, **_with_model_default(kwargs.get('model', dict())))
        self._local_model = QModel(observation_size, action_size, **_with_model_default(kwargs.get('model', dict())))
        self._gamma = kwargs.get('gamma', 0.99)
        self._tau = kwargs.get('tau', 0.1)

        self._print_config()

    @property
    def configuration(self):
        return {'observation_size': self._observation_size, 'action_size': self._action_size, **self._ctr_config}

    def _print_config(self):
        print(f"DQNAgent configuration:\n"
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

    def step(self, obs, action, reward, next_obs, done):
        if not isinstance(action, (collections.Sequence, np.ndarray)):
            action = [action]

        self._memory.record(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

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


def agent_save(agent, path):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / 'config.json', mode='w') as fp:
        json.dump(agent.configuration, fp)
    agent.save(path)


def agent_load(path):
    with open(path / 'config.json', mode='r') as fp:
        cfg = json.load(fp)
        agent = DQNAgent(**cfg)
        agent.load(path)
        return agent