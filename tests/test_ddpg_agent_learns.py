import pytest

from tests.auxiliary import GymSession
from udacity_rl.agents.ddpg_agent import DDPGAgent
from udacity_rl.epsilon import GaussianNoiseDecay


@pytest.fixture
def n_knob():
    return GymSession('gym_quickcheck:n-knob-v0', GaussianNoiseDecay(0.0, 1.0, 0.001, 0.999))


@pytest.fixture
def make_agent(n_knob):
    def factory(**kwargs):
        return DDPGAgent(n_knob.observation_space.shape[0], n_knob.action_space.shape[0], **kwargs)

    return factory


@pytest.fixture
def agent(make_agent):
    return make_agent()


@pytest.mark.stochastic(sample_size=10)
def test_untrained_agent_fails_at_n_knob(agent, n_knob, stochastic_run):
    stochastic_run.record(n_knob.test(agent))
    assert stochastic_run.average() == n_knob.reward_range[0]
