import gym
import pytest
from pytest import approx

from tests.auxiliary import GymSession, train_to_target
from udacity_rl.agents import DQNAgent, agent_save, agent_load
from udacity_rl.epsilon import EpsilonExpDecay


@pytest.fixture
def random_walk():
    return GymSession(gym.make('gym_quickcheck:random-walk-v0'), EpsilonExpDecay(1, 0.01, 0.999))


@pytest.fixture
def make_agent(random_walk):
    def factory(**kwargs):
        return DQNAgent(random_walk.observation_space, random_walk.action_space, **kwargs)

    return factory


@pytest.fixture
def agent(make_agent):
    return make_agent()


@pytest.mark.stochastic(sample_size=10)
def test_untrained_agent_fails_at_random_walk(agent, random_walk, stochastic_run):
    stochastic_run.record(random_walk.test(agent))
    assert stochastic_run.average() <= (random_walk.reward_range[0] + random_walk.reward_range[1]) / 2


@pytest.mark.stochastic(sample_size=10)
def test_agent_learns_random_walk(agent, random_walk, stochastic_run):
    random_walk.train(agent)
    stochastic_run.record(random_walk.test(agent))
    assert stochastic_run.average() == approx(random_walk.reward_range[1], abs=0.1)


@pytest.mark.stochastic(sample_size=10)
def test_agent_with_epsilon_one_is_as_bad_as_random(agent, random_walk, stochastic_run):
    random_walk.eps_calc = EpsilonExpDecay(1.0, 1.0, 1.0)
    random_walk.train(agent, train_freq=10)
    stochastic_run.record(random_walk.test(agent))
    assert stochastic_run.average() <= (random_walk.reward_range[0] + random_walk.reward_range[1]) / 2


def test_agent_can_be_saved_and_loaded(make_agent, random_walk, tmp_path):
    trained = make_agent()
    train_to_target(trained, random_walk, target_score=random_walk.reward_range[1] - 0.05)
    agent_save(trained, tmp_path / 'checkpoint/')

    loaded = agent_load(tmp_path / 'checkpoint/')

    assert random_walk.test(loaded) == approx(random_walk.reward_range[1], abs=0.1)
