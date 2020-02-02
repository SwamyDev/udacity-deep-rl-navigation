import contextlib
import json
from collections import deque
from pathlib import Path

import click
import gym
from unityagents import UnityEnvironment

from p1_navigation.adapter import GymAdapter
from p1_navigation.agent import DQNAgent


class UnityEnvFactory:
    def __init__(self, file_name):
        self._file_name = file_name

    def __call__(self):
        return GymAdapter(UnityEnvironment(str(self._file_name), no_graphics=True), brain_index=0)


class GymEnvFactory:
    def __init__(self, gym_name):
        self._gym_name = gym_name

    def __call__(self):
        return gym.make(self._gym_name)


@click.group()
@click.option('-e', '--environment', default=None, type=click.Path(dir_okay=False),
              help="path to the unity environment (default: resources/Banana_Linux/Banana.x86_64")
@click.option('-g', '--gym', default=None, type=click.STRING,
              help="name of a gym environment to train/test on (default: None)")
@click.pass_context
def cli(ctx, environment, gym):
    """
    CLI to train and run the navigation agent of the udacity project
    """
    env_fac = UnityEnvFactory(Path(__file__).absolute().parent.parent / "resources/Banana_Linux/Banana.x86_64")
    if environment:
        env_fac = UnityEnvFactory(environment)
    elif gym:
        env_fac = GymEnvFactory(gym)

    ctx.obj = dict(
        env_factory=env_fac
    )


class EpsilonCalc:
    def __init__(self, start, end, decay):
        self.eps = start
        self.end = end
        self.decay = decay

    def __call__(self):
        return self.eps

    def update(self):
        self.eps = max(self.eps * self.decay, self.end)


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
@click.pass_context
def train(ctx, episodes, config):
    cfg = dict()
    if config is not None:
        cfg = json.load(config)

    run_train_session(ctx.obj['env_factory'], episodes, cfg)


def run_train_session(env_fac, episodes, config):
    with environment_session(env_fac) as env:
        epsilon_fn = EpsilonCalc(config.get('eps_start', 1), config.get('eps_end', 0.01),
                                 config.get('eps_decay', 0.995))
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, epsilon_fn=epsilon_fn, **config)

        print(f"EpsilonCalc configuration:\n"
              f"\tstart:\t{epsilon_fn.eps}\n"
              f"\tend:\t{epsilon_fn.end}\n"
              f"\tdecay:\t{epsilon_fn.decay}\n")
        step = 0
        scores_last = deque(maxlen=100)
        scores_all = list()

        for episode in range(episodes):
            done = False
            score = 0
            obs = env.reset()
            while not done:
                action = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                agent.step(obs, action, reward, next_obs, done)
                obs = next_obs
                step += 1
                if step % 4 == 0:
                    agent.train()
                score += reward

            epsilon_fn.update()
            scores_last.append(score)
            scores_all.append(score)

            score_avg = sum(scores_last) / len(scores_last)
            reward_msg = f"\rEpisodes {episode}\tAverage reward: {score_avg :.2f}"
            print(reward_msg, end="")
            if episode % 100 == 0:
                print(reward_msg)

        return scores_all, score_avg


@cli.command()
@click.pass_context
def run(ctx):
    """
    run the trained agent on the specified environment
    """
    with environment_session(ctx.obj['env_factory']) as env:
        print(f"Number of actions: {env.action_space.n}")

        state = env.reset()
        print(f"State looks like: {state}")

        state_size = len(state)
        print(f"States have length: {state_size}")

        score = 0
        while True:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            score += reward
            state = next_state
            if done:
                break

        print(f"Score: {score}")


if __name__ == "__main__":
    cli()
