import pytest

from tests.auxiliary import GymSession, train_to_target
from udacity_rl.agents import agent_save, agent_load
from udacity_rl.agents.ddpg_agent import DDPGAgent
from udacity_rl.epsilon import EpsilonExpDecay


@pytest.fixture
def n_knob():
    return GymSession('gym_quickcheck:n-knob-v0', EpsilonExpDecay(1, 0.01, 0.995))


@pytest.fixture
def make_agent(n_knob):
    def factory(**kwargs):
        return DDPGAgent(n_knob.observation_space, n_knob.action_space, **kwargs)

    return factory


@pytest.fixture
def agent(make_agent):
    return make_agent()


@pytest.mark.stochastic(sample_size=5)
def test_untrained_agent_fails_at_n_knob(agent, n_knob, stochastic_run):
    stochastic_run.record(n_knob.test(agent))
    assert stochastic_run.average() <= -9.0


@pytest.mark.stochastic(sample_size=5)
def test_agent_learns_n_knob(agent, n_knob, stochastic_run):
    n_knob.train(agent, train_freq=2)
    stochastic_run.record(n_knob.test(agent))
    assert stochastic_run.average() >= -7.5


def test_agent_can_be_saved_and_loaded(make_agent, n_knob, tmp_path):
    trained = make_agent()
    train_to_target(trained, n_knob, target_score=-8)
    agent_save(trained, tmp_path / 'checkpoint/')

    loaded = agent_load(tmp_path / 'checkpoint/')

    assert n_knob.test(loaded) >= -8
