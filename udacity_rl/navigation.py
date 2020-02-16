import contextlib
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from pathlib import Path

import click
import gym
from unityagents import UnityEnvironment

from udacity_rl.adapter import GymAdapter
from udacity_rl.agent import DQNAgent, agent_load, agent_save
from udacity_rl.epsilon import EpsilonExpDecay


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


@click.group()
@click.option('-e', '--environment', default=None, type=click.Path(dir_okay=False),
              help="path to the unity environment (default: None")
@click.option('-g', '--gym', default=None, type=click.STRING,
              help="name of a gym environment to train/test on (default: CartPole-v0 )")
@click.pass_context
def cli(ctx, environment, gym):
    """
    CLI to train and run the navigation agent of the udacity project
    """
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
@click.argument('episodes', type=click.INT)
@click.option('-c', '--config', default=None, type=click.File(mode='r'),
              help="to training configuration file")
@click.option('-o', '--output', default="/tmp/p1_navigation_ckpt", type=click.Path(file_okay=False),
              help="path to store the agent at (default: /tmp/p1_navigation_ckpt)")
@click.pass_context
def train(ctx, episodes, config, output):
    cfg = dict()
    if config is not None:
        cfg = json.load(config)

    agent, scores = run_train_session(ctx.obj['env_factory'], episodes, cfg)
    agent_save(agent, Path(output))
    plot_scores(scores)


def run_train_session(env_fac, episodes, config):
    with environment_session(env_fac, train_mode=True) as env:
        eps_calc = EpsilonExpDecay(config.get('eps_start', 1), config.get('eps_end', 0.01),
                                   config.get('eps_decay', 0.995))
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, **config)

        print(f"Epsilon configuration:\n"
              f"\t{eps_calc}\n")
        scores = run_session(agent, env, episodes,
                             train_frequency=config.get('train_frequency', 4),
                             eps_calc=eps_calc)
        return agent, scores


def run_session(agent, env, episodes, train_frequency=None, eps_calc=None):
    step = 0
    scores_last = deque(maxlen=100)
    scores_all = list()
    for episode in range(episodes):
        done = False
        score = 0
        obs = env.reset()
        while not done:
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


if __name__ == "__main__":
    cli()
