import numpy as np
from gym import spaces


class GymAdapter:
    def __init__(self, unity_env, brain_index, train_mode=True):
        self._env = unity_env
        self._brain_index = brain_index
        self._train_mode = train_mode
        self._brain_name = self._env.brain_names[self._brain_index]
        self._brain = self._env.brains[self._brain_name]

        env_info = self._env.reset(train_mode=True)[self._brain_name]
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=self._squeeze_vector(env_info.vector_observations).shape)
        if self._brain.vector_action_space_type == 'discrete':
            self.action_space = spaces.Discrete(self._brain.vector_action_space_size)
        elif self._brain.vector_action_space_type == 'continuous':
            self.action_space = spaces.Box(-1.0, 1.0,
                                           shape=self._squeeze_vector(env_info.previous_vector_actions).shape)
        else:
            raise GymAdapterError(
                f"The action type '{self._brain.vector_action_space_type}' is currently not supported")

    @staticmethod
    def _squeeze_vector(vector):
        if len(vector) == 1:
            return vector[0]
        return vector

    @property
    def brain_name(self):
        return self._brain_name

    def step(self, action):
        info = self._env.step(action)[self._brain_name]
        obs, reward, done = info.vector_observations, info.rewards, any(info.local_done)
        return self._squeeze_vector(obs), self._squeeze_vector(reward), done, info

    def reset(self):
        info = self._env.reset(train_mode=self._train_mode)[self._brain_name]
        return self._squeeze_vector(info.vector_observations)

    def render(self, mode='human'):
        pass

    def close(self):
        self._env.close()


class GymAdapterError(ValueError):
    pass
