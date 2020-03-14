import gym
import numpy as np
import pytest
from gym import Wrapper
from gym.spaces import Box

from tests.auxiliary import GymSession
from udacity_rl.agents import agent_save, agent_load
from udacity_rl.agents.maddpg_agent import MADDPGAgent
from udacity_rl.epsilon import NoiseFixed


class MultiAgentEnvWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(-np.inf, np.inf, shape=(1, env.observation_space.shape[0]))
        self.action_space = Box(-1.0, 1.0, shape=(1, env.action_space.shape[0]))

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self._wrap_observation(observation)

    def step(self, action):
        # noinspection PyTypeChecker
        observation, reward, done, info = self.env.step(self._wrap_action(action))
        return self._wrap_observation(observation), self._wrap_reward(reward), done, info

    @staticmethod
    def _wrap_observation(observation):
        return observation[np.newaxis, :]

    @staticmethod
    def _wrap_action(action):
        return action[0]

    @staticmethod
    def _wrap_reward(reward):
        return np.array([reward])


@pytest.fixture
def n_knob():
    return GymSession(MultiAgentEnvWrapper(gym.make('gym_quickcheck:n-knob-v0')), NoiseFixed(0.01))


@pytest.fixture
def make_agent(n_knob):
    def factory(**kwargs):
        return MADDPGAgent(n_knob.observation_space, n_knob.action_space, **kwargs)

    return factory


@pytest.fixture
def agent(make_agent):
    return make_agent()


@pytest.mark.stochastic(sample_size=5)
def test_untrained_agent_fails_at_n_knob(agent, n_knob, stochastic_run):
    stochastic_run.record(n_knob.test(agent))
    assert stochastic_run.average() <= -9.0


@pytest.mark.stochastic(sample_size=5)
@pytest.mark.flaky(reruns=3)
def test_agent_learns_n_knob(agent, n_knob, stochastic_run):
    n_knob.train(agent, train_freq=2)
    stochastic_run.record(n_knob.test(agent))
    assert stochastic_run.average() >= -7.5


def test_save_and_load_agent(agent, tmp_path):
    agent_save(agent, tmp_path / 'checkpoint/')
    loaded = agent_load(tmp_path / 'checkpoint/')
    assert loaded is not None
