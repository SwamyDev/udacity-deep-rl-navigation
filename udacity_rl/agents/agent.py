import abc
import collections
import numpy as np

from gym import spaces

from udacity_rl.memory import Memory, UniformReplayBuffer


def with_default(cfg, key, default):
    if key not in cfg:
        cfg[key] = default
    return cfg


class AgentInterface(abc.ABC):
    class ConfigurationError(ValueError):
        pass

    @property
    @abc.abstractmethod
    def action_size(self):
        pass

    @property
    @abc.abstractmethod
    def observation_space(self):
        pass

    @property
    @abc.abstractmethod
    def action_space(self):
        pass

    @property
    @abc.abstractmethod
    def configuration(self):
        pass

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


class Agent(AgentInterface, abc.ABC):
    def __init__(self, observation_space, action_space, **kwargs):
        self._ctr_config = kwargs
        self._observation_space = observation_space
        self._action_space = action_space
        self._observation_size = self._get_size_from_space(self.observation_space)
        self._action_size = self._get_size_from_space(self.action_space)

    @staticmethod
    def _get_size_from_space(space):
        if isinstance(space, spaces.Discrete):
            return space.n
        elif isinstance(space, spaces.Box):
            return space.shape[0]
        else:
            raise AgentInterface.ConfigurationError(f"the space {space} is currently not supported")

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def action_size(self):
        return self._action_size

    @property
    def configuration(self):
        return self._ctr_config


def _with_mem_defaults(cfg):
    cfg = with_default(cfg, 'batch_size', 64)
    cfg = with_default(cfg, 'record_size', int(1e5))
    cfg = with_default(cfg, 'seed', None)
    return cfg


class MemoryAgent(Agent, abc.ABC):
    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        mem_cfg = _with_mem_defaults(kwargs)
        self._memory = Memory(mem_cfg["batch_size"], UniformReplayBuffer(mem_cfg["record_size"], mem_cfg["seed"]))

    def step(self, obs, action, reward, next_obs, done):
        if not isinstance(action, (collections.Sequence, np.ndarray)):
            action = [action]

        self._memory.record(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
