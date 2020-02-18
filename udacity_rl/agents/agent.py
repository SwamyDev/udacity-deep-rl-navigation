import abc


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
