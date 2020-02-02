from pytest import approx

from p1_navigation.navigation import run_train_session, GymEnvFactory


def test_navigation_training_invocation():
    _, score_last_avg = run_train_session(GymEnvFactory('gym_quickcheck:random-walk-v0'), 1000, dict())
    assert score_last_avg == approx(-1, abs=0.3)
