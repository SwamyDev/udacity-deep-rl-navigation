import pytest

from p1_navigation.epsilon import EpsilonExpDecay


@pytest.fixture
def make_decay():
    def factory(*args, **kwargs):
        return EpsilonExpDecay(*args, **kwargs)

    return factory


@pytest.mark.parametrize('start', (1.0, 2.0))
def test_epsilon_exponential_decay_start_value(make_decay, start):
    decay = make_decay(start, end=0.01, rate=0.9)
    assert decay.epsilon == start


def test_epsilon_exponential_decay_decreases(make_decay):
    decay = make_decay(start=1.0, end=0.01, rate=0.9)
    decay.update()
    assert decay.epsilon == 0.9
    decay.update()
    assert decay.epsilon == 0.9 * 0.9


def test_epsilon_exponential_decay_stops_at_minimum_value(make_decay):
    decay = make_decay(start=1.0, end=0.01, rate=0.1)
    for _ in range(3):
        decay.update()
    assert decay.epsilon == 0.01


def test_epsilon_exponential_decay_string_representation(make_decay):
    s = repr(make_decay(start=1.0, end=0.01, rate=0.9))
    assert "EpsilonExpDecay" in s and "1.0" in s and "0.01" in s and "0.9" in s
