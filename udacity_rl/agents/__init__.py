import json
import logging
import pickle

from udacity_rl.agents.dqn_agent import DQNAgent
from udacity_rl.agents.ddpg_agent import DDPGAgent
from udacity_rl.agents.nddpg_agent import NDDPGAgent

logger = logging.getLogger(__name__)

_CLASS_MAPPING = {
    DQNAgent.__name__: DQNAgent,
    DDPGAgent.__name__: DDPGAgent,
    NDDPGAgent.__name__: NDDPGAgent,
}


def agent_save(agent, path):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / 'type.meta', mode='w') as fp:
        fp.write(type(agent).__name__)
    with open(path / 'observation_space', mode='wb') as fp:
        pickle.dump(agent.observation_space, fp)
    with open(path / 'action_space', mode='wb') as fp:
        pickle.dump(agent.action_space, fp)
    with open(path / 'config.json', mode='w') as fp:
        json.dump(agent.configuration, fp)
    agent.save(path)


def agent_load(path):
    with open(path / 'type.meta', mode='r') as fp:
        agent_type = fp.read()
    with open(path / 'observation_space', mode='rb') as fp:
        obs_space = pickle.load(fp)
    with open(path / 'action_space', mode='rb') as fp:
        act_space = pickle.load(fp)
    with open(path / 'config.json', mode='r') as fp:
        cfg = json.load(fp)
    agent = _CLASS_MAPPING[agent_type](obs_space, act_space, **cfg)
    agent.load(path)
    return agent


class AgentSnapshot:
    def __init__(self, agent, target_score, path):
        self._agent = agent
        self._target = target_score
        self._path = path

    def new_score(self, score):
        if self._target is not None and score > self._target:
            logger.info(f"saving agent ({self._path}) snapshot with score: {score}")
            agent_save(self._agent, self._path)
            self._target = score
            logger.info(f"new save threshold: {self._target}")
