import contextlib
from numbers import Number
from pathlib import Path

import pytest
from unityagents import UnityEnvironment

from p1_navigation.adapter import GymAdapter
from tests.auxiliary import assert_that, follows_contract


@pytest.fixture(scope='module')
def unity_env():
    with loaded_unity_env(
            file_name=Path(__file__).absolute().parent.parent / "resources/Banana_Linux/Banana.x86_64") as env:
        yield env


@contextlib.contextmanager
def loaded_unity_env(file_name):
    env = UnityEnvironment(str(file_name), no_graphics=True)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture(scope='session')
def gym_interface():
    return [('reset', ()), ('step', (0,)), ('render', ())]


@pytest.fixture(scope='session')
def gym_properties():
    return ['action_space', 'observation_space']


@pytest.fixture
def adapter(unity_env):
    return GymAdapter(unity_env, brain_index=0)


@pytest.mark.unity
def test_gym_adapter_adheres_to_gym_contract(adapter, gym_interface, gym_properties):
    assert_that(adapter, follows_contract(gym_interface, gym_properties))


@pytest.mark.unity
def test_uses_specified_brain(adapter):
    assert adapter.brain_name == "BananaBrain"


@pytest.mark.unity
def test_has_matching_action_and_observation_spaces(adapter):
    assert adapter.action_space.n == 4 and adapter.observation_space.shape == (37,)


@pytest.mark.unity
def test_reset_returns_observation_of_selected_brain(adapter):
    assert adapter.reset().shape == (37,)


@pytest.mark.unity
def test_step_returns_next_observation_reward_done_and_info(adapter):
    obs, reward, done, info = adapter.step(0)
    assert obs.shape == (37,) and isinstance(reward, Number) and isinstance(done, type(True)) and info is not None
