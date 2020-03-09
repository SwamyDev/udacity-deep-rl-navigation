import json
import pickle

from udacity_rl.agents.dqn_agent import DQNAgent
from udacity_rl.agents.ddpg_agent import DDPGAgent
from udacity_rl.agents.maddpg_agent import MADDPGAgent

_CLASS_MAPPING = {
    DQNAgent.__name__: DQNAgent,
    DDPGAgent.__name__: DDPGAgent,
    MADDPGAgent.__name__: MADDPGAgent,
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
