import pytest
import numpy as np
from gym import ObservationWrapper, ActionWrapper
from gym.spaces import Box

from tests.auxiliary import GymSession
from udacity_rl.agents.maddpg_agent import MADDPGAgent
from udacity_rl.epsilon import EpsilonExpDecay


class MultiAgentEnvWrapper(ObservationWrapper, ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(-np.inf, np.inf, shape=(1, env.observation_space.shape[0]))
        self.action_space = Box(-1.0, 1.0, shape=(1, env.action_space.shape[0]))

    def observation(self, observation):
        return observation[np.newaxis, :]

    def action(self, action):
        return action[0]

    def reverse_action(self, action):
        return action[np.newaxis, :]


@pytest.fixture
def n_knob():
    return MultiAgentEnvWrapper(GymSession('gym_quickcheck:n-knob-v0', EpsilonExpDecay(1, 0.01, 0.995)))


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
