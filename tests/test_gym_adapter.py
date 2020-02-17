import contextlib
import numpy as np
from numbers import Number
from pathlib import Path

import pytest
from gym.spaces import Discrete, Box
from unityagents import UnityEnvironment

from udacity_rl.adapter import GymAdapter
from tests.auxiliary import assert_that, follows_contract


@pytest.fixture(scope='module')
def unity_env(use_reacher):
    env_file = "resources/environments/Banana_Linux/Banana.x86_64"
    if use_reacher:
        env_file = "resources/environments/Reacher_Linux/Reacher.x86_64"
    with loaded_unity_env(file_name=Path(__file__).absolute().parent.parent / env_file) as env:
        yield env


@contextlib.contextmanager
def loaded_unity_env(file_name):
    env = UnityEnvironment(str(file_name), no_graphics=True)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture(scope='session')
def gym_interface(use_reacher):
    action = (0,)
    if use_reacher:
        action = ([0, 0, 0, 0],)
    return [('reset', ()), ('step', action), ('render', ())]


@pytest.fixture(scope='session')
def gym_properties():
    return ['action_space', 'observation_space']


@pytest.fixture
def adapter(unity_env):
    return GymAdapter(unity_env, brain_index=0)


@pytest.fixture
def unity_brain(use_reacher):
    if use_reacher:
        return "ReacherBrain"
    return "BananaBrain"


@pytest.fixture
def unity_actions(use_reacher):
    if use_reacher:
        return Box(-1.0, 1.0, shape=(4,))
    return Discrete(4)


@pytest.fixture
def unity_observation_shape(use_reacher):
    if use_reacher:
        return (33,)
    return (37,)


def test_gym_adapter_adheres_to_gym_contract(adapter, gym_interface, gym_properties):
    assert_that(adapter, follows_contract(gym_interface, gym_properties))


def test_uses_specified_brain(adapter, unity_brain):
    assert adapter.brain_name == unity_brain


def test_has_matching_action_and_observation_spaces(adapter, unity_actions, unity_observation_shape):
    assert adapter.action_space == unity_actions and adapter.observation_space.shape == unity_observation_shape


def test_reset_returns_observation_of_selected_brain(adapter, unity_observation_shape):
    assert adapter.reset().shape == unity_observation_shape


def test_step_returns_next_observation_reward_done_and_info(adapter, unity_actions, unity_observation_shape):
    obs, reward, done, info = adapter.step(unity_actions.sample())
    assert obs.shape == unity_observation_shape and \
           isinstance(reward, Number) and \
           isinstance(done, type(True)) and \
           info is not None
