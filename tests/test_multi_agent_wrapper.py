from collections import namedtuple
from pathlib import Path

import pytest
import numpy as np
from gym.spaces import Box

from udacity_rl.agents.agent import MemoryAgent, MultiAgentWrapper, AgentInterface, Agent


class AgentStub(MemoryAgent):
    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self.return_action = None

    def act(self, observation, epsilon=0):
        return self.return_action

    def train(self):
        pass

    def save(self, save_path):
        pass

    def load(self, save_path):
        pass


class AgentSpy(AgentStub):
    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self.received_obs = []
        self.received_eps = []
        self.received_step = []
        self._step = namedtuple("Step", ["obs", "action", "reward", "next_obs", "done"])
        self.received_train = 0
        self.received_save_path = None
        self.received_load_path = None

    def act(self, observation, epsilon=0):
        self.received_obs.append(observation)
        self.received_eps.append(epsilon)
        return super().act(observation, epsilon)

    def step(self, obs, action, reward, next_obs, done):
        super().step(obs, action, reward, next_obs, done)
        self.received_step.append(self._step(obs, action, reward, next_obs, done))

    def train(self):
        super().train()
        self.received_train += 1

    def save(self, save_path):
        super().save(save_path)
        self.received_save_path = save_path

    def load(self, save_path):
        super().load(save_path)
        self.received_load_path = save_path


def make_tensor(value):
    return np.array(value)


@pytest.fixture
def make_obs():
    return make_tensor


@pytest.fixture
def make_act(make_obs):
    return make_obs


@pytest.fixture
def make_reward(make_obs):
    return make_obs


@pytest.fixture
def make_wrapped_agent():
    def factory():
        return AgentSpy(Box(-1.0, 1.0, shape=(3,)), Box(-1.0, 1.0, shape=(2,)))

    return factory


@pytest.fixture
def wrapped_agents(make_wrapped_agent):
    return [make_wrapped_agent(), make_wrapped_agent()]


@pytest.fixture
def agent(wrapped_agents):
    return MultiAgentWrapper(wrapped_agents)


def test_multi_agent_wrapper_implements_agent_interface(agent):
    assert issubclass(type(agent), AgentInterface)


def test_raise_an_error_when_observation_and_num_agents_mismatch(agent, make_obs):
    with pytest.raises(MultiAgentWrapper.ConfigurationError):
        agent.act(make_obs([[1, 2, 3]]), 0)


def assert_tensor(actual, expected):
    np.testing.assert_array_equal(actual, expected)


def test_distribute_observation_to_each_agent(agent, make_obs, wrapped_agents):
    agent.act(make_obs([[1, 1, 1], [2, 2, 2]]), 0)
    assert_tensor(wrapped_agents[0].received_obs[0], make_obs([1, 1, 1]))
    assert_tensor(wrapped_agents[1].received_obs[0], make_obs([2, 2, 2]))


def test_pass_on_epsilon_to_wrapped_agents(agent, make_obs, wrapped_agents):
    agent.act(make_obs([[1, 1, 1], [2, 2, 2]]), 0.5)
    assert all(a.received_eps[0] == 0.5 for a in wrapped_agents)


def test_fuse_actions_of_wrapped_agents_to_one_tensor(agent, make_obs, make_act, wrapped_agents):
    wrapped_agents[0].return_action = make_act([0.2, 0.8])
    wrapped_agents[1].return_action = make_act([0.6, 0.4])
    assert_tensor(agent.act(make_obs([[1, 1, 1], [2, 2, 2]]), 0), make_act([[0.2, 0.8], [0.6, 0.4]]))


def test_step_is_passed_to_wrapped_agents(agent, make_obs, make_act, make_reward, wrapped_agents):
    agent.step(make_obs([[1, 1, 1], [2, 2, 2]]), make_act([[0.2, 0.8], [0.6, 0.4]]), make_reward([-1.0, 1.0]),
               make_obs([[3, 3, 3], [4, 4, 4]]), True)
    assert_tensor(wrapped_agents[0].received_step[0].obs, make_obs([1, 1, 1]))
    assert_tensor(wrapped_agents[1].received_step[0].obs, make_obs([2, 2, 2]))
    assert_tensor(wrapped_agents[0].received_step[0].action, make_obs([0.2, 0.8]))
    assert_tensor(wrapped_agents[1].received_step[0].action, make_obs([0.6, 0.4]))
    assert wrapped_agents[0].received_step[0].reward == -1.0
    assert wrapped_agents[1].received_step[0].reward == 1.0
    assert_tensor(wrapped_agents[0].received_step[0].next_obs, make_obs([3, 3, 3]))
    assert_tensor(wrapped_agents[1].received_step[0].next_obs, make_obs([4, 4, 4]))
    assert wrapped_agents[0].received_step[0].done and wrapped_agents[1].received_step[0].done


@pytest.mark.parametrize("obs, act, reward, next_obs", [
    (make_tensor([0, 0, 0]), make_tensor([[0, 0], [0, 0]]), [0, 0], make_tensor([[0, 0, 0], [0, 0, 0]])),
    (make_tensor([[0, 0, 0], [0, 0, 0]]), make_tensor([[0, 0]]), [0, 0], make_tensor([[0, 0, 0], [0, 0, 0]])),
    (make_tensor([[0, 0, 0], [0, 0, 0]]), make_tensor([[0, 0], [0, 0]]), [0], make_tensor([[0, 0, 0], [0, 0, 0]])),
    (make_tensor([[0, 0, 0], [0, 0, 0]]), make_tensor([[0, 0], [0, 0]]), [0, 0], make_tensor([[0, 0, 0]]))
])
def test_raise_an_error_when_step_parameters_and_num_agents_mismatch(agent, obs, act, reward, next_obs):
    with pytest.raises(MultiAgentWrapper.ConfigurationError):
        agent.step(obs, act, reward, next_obs, True)


def test_train_is_passed_to_wrapped_agents(agent, wrapped_agents):
    agent.train()
    assert all(a.received_train for a in wrapped_agents)


def test_pass_on_save_path_with_agent_suffix(agent, wrapped_agents, tmp_path):
    agent.save(tmp_path)
    assert wrapped_agents[0].received_save_path == tmp_path / "wrapped_agent_0"
    assert wrapped_agents[1].received_save_path == tmp_path / "wrapped_agent_1"
    assert (tmp_path / "wrapped_agent_0").exists()
    assert (tmp_path / "wrapped_agent_1").exists()


def test_pass_on_load_path_with_agent_suffix(agent, wrapped_agents, tmp_path):
    agent.load(tmp_path)
    assert wrapped_agents[0].received_load_path == tmp_path / "wrapped_agent_0"
    assert wrapped_agents[1].received_load_path == tmp_path / "wrapped_agent_1"
