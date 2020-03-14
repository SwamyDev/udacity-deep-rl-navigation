import pytest
from udacity_rl.agents import AgentSnapshot
from udacity_rl.agents.agent import AgentInterface


class AgentStub(AgentInterface):
    @property
    def action_size(self):
        pass

    @property
    def observation_space(self):
        pass

    @property
    def action_space(self):
        pass

    @property
    def configuration(self):
        pass

    def act(self, observation, epsilon=0):
        pass

    def step(self, obs, action, reward, next_obs, done):
        pass

    def train(self):
        pass

    def save(self, save_path):
        pass

    def load(self, save_path):
        pass


class AgentSpy(AgentStub):
    def __init__(self):
        self.received_save_path = None
        self.num_save_calls = 0

    def save(self, save_path):
        self.received_save_path = save_path
        self.num_save_calls += 1
        super().save(save_path)


@pytest.fixture
def agent():
    return AgentSpy()


def test_agent_snapshot_does_nothing_if_new_score_is_below_target(agent, tmp_path):
    snapshot = AgentSnapshot(agent, 10, tmp_path / "snapshot")
    snapshot.new_score(10)
    assert agent.received_save_path is None


def test_agent_snapshot_saves_agent_if_new_score_is_above_target(agent, tmp_path):
    snapshot = AgentSnapshot(agent, 10, tmp_path / "snapshot")
    snapshot.new_score(11)
    assert agent.received_save_path == tmp_path / "snapshot"


def test_agent_snapshot_updates_new_target(agent, tmp_path):
    snapshot = AgentSnapshot(agent, 10, tmp_path / "snapshot")
    snapshot.new_score(11)
    assert agent.received_save_path == tmp_path / "snapshot"
    snapshot.new_score(11)
    assert agent.num_save_calls == 1


def test_do_nothing_if_target_is_none(agent, tmp_path):
    snapshot = AgentSnapshot(agent, None, tmp_path / "snapshot")
    snapshot.new_score(11)
    assert agent.num_save_calls == 0
