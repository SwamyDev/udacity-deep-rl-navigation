import json
from collections import deque
from pathlib import Path

import click
import numpy as np
from unityagents import UnityEnvironment

from p1_navigation.agent import DQNAgent


@click.group()
@click.option('-e', '--environment', default=None, type=click.Path(dir_okay=False),
              help="path to the unity environment (default: resources/Banana_Linux/Banana.x86_64")
@click.pass_context
def cli(ctx, environment):
    """
    CLI to train and run the navigation agent of the udacity project
    """
    ctx.obj = dict(
        env_file=environment or (Path(__file__).absolute().parent.parent / "resources/Banana_Linux/Banana.x86_64")
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


@cli.command()
@click.argument('episodes', type=click.INT)
@click.option('-c', '--config', default=None, type=click.File(mode='r'),
              help="to training configuration file")
@click.pass_context
def train(ctx, episodes, config):
    cfg = dict()
    if config is not None:
        cfg = json.load(config)

    env = UnityEnvironment(file_name=str(ctx.obj['env_file']), no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    observation_size = brain.vector_observation_space_size
    action_size = brain.vector_action_space_size
    epsilon_fn = EpsilonCalc(cfg.get('eps_start', 1), cfg.get('eps_end', 0.01), cfg.get('eps_decay', 0.995))
    agent = DQNAgent(observation_size, action_size, epsilon_fn=epsilon_fn, **cfg)

    print(f"EpsilonCalc configuration:\n"
          f"\tstart:\t{epsilon_fn.eps}\n"
          f"\tend:\t{epsilon_fn.end}\n"
          f"\tdecay:\t{epsilon_fn.decay}\n")
    step = 0
    scores_last = deque(maxlen=100)
    scores_all = list()

    for episode in range(episodes):
        done = False
        info = env.reset(train_mode=True)[brain_name]
        obs = info.vector_observations[0]
        score = 0
        while not done:
            action = agent.act(obs)
            info = env.step(action)[brain_name]
            next_obs, reward, done = info.vector_observations[0], info.rewards[0], info.local_done[0]
            agent.step(obs, action, reward, next_obs, done)
            obs = next_obs
            step += 1
            if step % 4 == 0:
                agent.train()
            score += reward

        epsilon_fn.update()
        scores_last.append(score)
        scores_all.append(score)

        reward_msg = f"\rEpisodes {episode}\tAverage reward: {sum(scores_last) / len(scores_last) :.2f}"
        print(reward_msg, end="")
        if episode % 100 == 0:
            print(reward_msg)


@cli.command()
@click.pass_context
def run(ctx):
    """
    run the trained agent on the specified environment
    """
    env = UnityEnvironment(file_name=str(ctx.obj['env_file']))
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    print(f"Number of agents: {len(env_info.agents)}")

    action_size = brain.vector_action_space_size
    print(f"Number of actions: {action_size}")

    state = env_info.vector_observations[0]
    print(f"State looks like: {state}")

    state_size = len(state)
    print(f"States have length: {state_size}")

    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    while True:
        action = np.random.randint(action_size)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            break

    print(f"Score: {score}")


if __name__ == "__main__":
    cli()
