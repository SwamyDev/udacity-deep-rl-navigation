import numpy as np
from pytest import approx

from udacity_rl.navigation import run_train_session, GymEnvFactory, run_test_session


def test_navigation_training_invocation():
    _, scores = run_train_session(GymEnvFactory('gym_quickcheck:random-walk-v0'), 1000, dict())
    assert np.mean(scores[-100:]) == approx(-1, abs=0.3)


def test_navigation_run_invocation():
    agent, _ = run_train_session(GymEnvFactory('gym_quickcheck:random-walk-v0'), 1000, dict())
    scores = run_test_session(agent, GymEnvFactory('gym_quickcheck:random-walk-v0'), 100)
    assert np.mean(scores) == approx(-1, abs=0.3)
