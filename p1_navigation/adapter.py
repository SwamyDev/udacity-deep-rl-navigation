import numpy as np
from gym import spaces


class GymAdapter:
    def __init__(self, unity_env, brain_index, train_mode=True):
        self._env = unity_env
        self._brain_index = brain_index
        self._train_mode = train_mode
        self._brain_name = self._env.brain_names[self._brain_index]
        self._brain = self._env.brains[self._brain_name]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self._brain.vector_observation_space_size,))
        self.action_space = spaces.Discrete(self._brain.vector_action_space_size)

    @property
    def brain_name(self):
        return self._brain_name

    def step(self, action):
        info = self._env.step(action)[self._brain_name]
        idx = self._brain_index
        obs, reward, done = info.vector_observations[idx], info.rewards[idx], info.local_done[idx]
        return obs, reward, done, info

    def reset(self):
        info = self._env.reset(train_mode=self._train_mode)[self._brain_name]
        return info.vector_observations[self._brain_index]

    def render(self, mode='human'):
        pass

    def close(self):
        self._env.close()
