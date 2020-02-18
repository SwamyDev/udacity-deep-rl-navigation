import json
from udacity_rl.agents.dqn_agent import DQNAgent


def agent_save(agent, path):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / 'config.json', mode='w') as fp:
        json.dump(agent.configuration, fp)
    agent.save(path)


def agent_load(path):
    with open(path / 'config.json', mode='r') as fp:
        cfg = json.load(fp)
        agent = DQNAgent(**cfg)
        agent.load(path)
        return agent
