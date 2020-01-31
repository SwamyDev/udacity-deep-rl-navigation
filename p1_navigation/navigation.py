from pathlib import Path

import click
import numpy as np
from unityagents import UnityEnvironment


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
