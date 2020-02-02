import gym
import pytest
from pytest import approx

from p1_navigation.agent import DQNAgent


class RandomWalkSession(gym.Wrapper):
    def __init__(self):
        super().__init__(gym.make('gym_quickcheck:random-walk-v0'))

    def test(self, agent, num_episodes=100):
        return self._run_session(agent, num_episodes, is_test=True)

    def train(self, agent, num_episodes=1000):
        return self._run_session(agent, num_episodes, is_test=False)

    def _run_session(self, agent, num_episodes, is_test=False):
        average_reward = 0
        for _ in range(num_episodes):
            done = False
            obs = self.env.reset()
            total_r = 0
            while not done:
                a = agent.act(obs)
                next_obs, r, done, _ = self.env.step(a)
                agent.step(obs, a, r, next_obs, done)
                obs = next_obs
                total_r += r
            average_reward += total_r
            if not is_test:
                agent.train()

        return average_reward / num_episodes


@pytest.fixture
def random_walk():
    return RandomWalkSession()


@pytest.fixture
def make_agent(random_walk):
    def factory(**kwargs):
        return DQNAgent(random_walk.observation_space.shape[0], random_walk.action_space.n, **kwargs)

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
def test_agent_with_epsilon_one_is_as_bad_as_random(make_agent, random_walk, stochastic_run):
    agent = make_agent(epsilon_fn=lambda: 1)
    random_walk.train(agent)
    stochastic_run.record(random_walk.test(agent))
    assert stochastic_run.average() <= (random_walk.reward_range[0] + random_walk.reward_range[1]) / 2
