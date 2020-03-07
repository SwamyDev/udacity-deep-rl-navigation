import contextlib
import json
import matplotlib.pyplot as plt
import numpy as np
import logging
from collections import deque
from pathlib import Path

import click
import gym
from unityagents import UnityEnvironment

from udacity_rl.adapter import GymAdapter
from udacity_rl.agents import DQNAgent, agent_load, agent_save
from udacity_rl.agents.ddpg_agent import DDPGAgent
from udacity_rl.epsilon import EpsilonExpDecay

logger = logging.getLogger(__name__)


class UnityEnvFactory:
    def __init__(self, file_name):
        self._file_name = file_name

    def __call__(self, *args, train_mode=True, render=False, **kwargs):
        return GymAdapter(UnityEnvironment(str(self._file_name), no_graphics=not render), brain_index=0,
                          train_mode=train_mode)


class GymEnvFactory:
    def __init__(self, gym_name):
        self._gym_name = gym_name

    def __call__(self, *args, **kwargs):
        return gym.make(self._gym_name)


class AgentFactory:
    _AGENT_MAPPING = {
        'DQN': DQNAgent,
        'DDPG': DDPGAgent,
    }

    def __init__(self, algorithm_name):
        self._algorithm_name = algorithm_name

    def __call__(self, *args, **kwargs):
        return self._AGENT_MAPPING[self._algorithm_name](*args, **kwargs)


@click.group()
@click.option('-e', '--environment', default=None, type=click.Path(dir_okay=False),
              help="path to the unity environment (default: None")
@click.option('-g', '--gym', default=None, type=click.STRING,
              help="name of a gym environment to train/test on (default: CartPole-v0 )")
@click.option('--log-level', default="INFO", type=click.STRING,
              help="set the logging level (default: INFO)")
@click.pass_context
def cli(ctx, environment, gym, log_level):
    """
    CLI to train and run the navigation agent of the udacity project
    """
    numeric_level = getattr(logging, log_level.upper())
    logging.basicConfig(level=numeric_level)
    logging.root.setLevel(numeric_level)

    if environment:
        env_fac = UnityEnvFactory(environment)
    elif gym:
        env_fac = GymEnvFactory(gym)
    else:
        env_fac = GymEnvFactory('CartPole-v0')

    ctx.obj = dict(
        env_factory=env_fac
    )


@contextlib.contextmanager
def environment_session(env_factory, *args, **kwargs):
    env = env_factory(*args, **kwargs)
    try:
        yield env
    finally:
        env.close()


@cli.command()
@click.argument('algorithm', type=click.STRING)
@click.argument('episodes', type=click.INT)
@click.option('-c', '--config', default=None, type=click.File(mode='r'),
              help="to training configuration file")
@click.option('-o', '--output', default="/tmp/agent_ckpt", type=click.Path(file_okay=False),
              help="path to store the agent at (default: /tmp/agent_ckpt)")
@click.option('--max-t', default=None, type=click.INT,
              help="maximum episode steps (default: None)")
@click.pass_context
def train(ctx, algorithm, episodes, config, output, max_t):
    """
    train the agent with the specified algorithm on the environment for the given amount of episodes
    """
    cfg = dict()
    if config is not None:
        cfg = json.load(config)

    agent, scores = run_train_session(ctx.obj['env_factory'], AgentFactory(algorithm), episodes, cfg, max_t)
    agent_save(agent, Path(output))
    plot_scores(scores)


def run_train_session(env_fac, agent_fac, episodes, config, max_t):
    with environment_session(env_fac, train_mode=True) as env:
        eps_calc = EpsilonExpDecay(config.get('eps_start', 1), config.get('eps_end', 0.01),
                                   config.get('eps_decay', 0.995))
        agent = agent_fac(env.observation_space, env.action_space, **config)

        logger.info(f"Epsilon configuration:\n"
                    f"\t{eps_calc}\n")
        scores = run_session(agent, env, episodes,
                             train_frequency=config.get('train_frequency', 4),
                             eps_calc=eps_calc,
                             max_t=max_t)
        return agent, scores


def run_session(agent, env, episodes, train_frequency=None, eps_calc=None, max_t=None):
    step = 0
    scores_last = deque(maxlen=100)
    scores_all = list()
    for episode in range(episodes):
        done = False
        score = 0
        obs = env.reset()
        t = 0
        while not done and (max_t is None or t < max_t):
            action = agent.act(obs, 0 if eps_calc is None else eps_calc.epsilon)
            next_obs, reward, done, _ = env.step(action)
            agent.step(obs, action, reward, next_obs, done)
            obs = next_obs
            step += 1
            if train_frequency is not None and step % train_frequency == 0:
                agent.train()
            score += reward

        if eps_calc:
            eps_calc.update()
        scores_last.append(score)
        scores_all.append(score)

        score_avg = sum(scores_last) / len(scores_last)
        reward_msg = f"\rEpisodes ({episode}/{episodes})\tAverage reward: {score_avg :.2f}"
        print(reward_msg, end="")
        if episode % 100 == 0:
            print(reward_msg)
    return scores_all


def run_test_session(agent, env_fac, episodes, render=False):
    with environment_session(env_fac, train_mode=not render, render=render) as env:
        return run_session(agent, env, episodes)


@cli.command()
@click.argument('agent', type=click.Path(file_okay=False))
@click.argument('episodes', type=click.INT)
@click.option('-r', '--render', default=True, type=click.BOOL,
              help="render the environment (default: true)")
@click.pass_context
def run(ctx, agent, episodes, render):
    """
    run the trained agent on the specified environment
    """
    scores = run_test_session(agent_load(Path(agent)), ctx.obj['env_factory'], episodes, render)
    plot_scores(scores, 5)


def plot_scores(scores, avg_window=100):
    scores_avg = moving_average(scores, avg_window)
    fig, ax = plt.subplots()
    ax.plot(range(len(scores)), scores)
    start = avg_window // 2
    ax.plot(range(start, start + len(scores_avg)), scores_avg)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def moving_average(a, n=3):
    ret = np.cumsum(np.insert(a, 0, 0))
    return (ret[n:] - ret[:-n]) / n


@cli.command()
@click.option('-n', '--num', default=1, type=click.INT,
              help="number (-1 infinite runs) of episodes to explore. (default: 1)")
@click.pass_context
def explore(ctx, num):
    """
    explore the specified environment by logging observation and action spaces and rendering an episode
    """
    with environment_session(ctx.obj['env_factory'], train_mode=False, render=True) as env:
        logger.info(f'Observation space: {env.observation_space}')
        logger.info(f'Action space: {env.action_space}')
        e = 0
        while e < num or num == -1:
            done = False
            env.reset()
            while not done:
                _, _, done, _ = env.step(env.action_space.sample())
            e += 1


if __name__ == "__main__":
    cli()
