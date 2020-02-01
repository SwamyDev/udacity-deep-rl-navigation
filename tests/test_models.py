import pytest
from pytest import approx

from p1_navigation.model import QModel


@pytest.fixture
def make_model():
    def factory(obs_size=3, act_size=2, layers=None):
        layers = layers or [dict(activation='relu', size=64)]
        return QModel(obs_size, act_size, layers)

    return factory


@pytest.fixture
def model(make_model):
    return make_model()


def test_estimates_have_expected_shape(model):
    assert model.estimate([[0, 0, 0]]).shape == (1, 2)


@pytest.mark.flaky(reruns=3)
def test_fitting_observations_to_target_values(model):
    assert model.estimate([[0, 0, 0]]).squeeze()[0] != approx(1, abs=0.001)
    assert model.estimate([[1, 1, 1]]).squeeze()[1] != approx(-1, abs=0.001)

    train(model, episodes=1000)

    assert model.estimate([[0, 0, 0]]).squeeze()[0] == approx(1, abs=0.001)
    assert model.estimate([[1, 1, 1]]).squeeze()[1] == approx(-1, abs=0.001)


def train(model, episodes):
    for _ in range(episodes):
        model.fit([[0, 0, 0], [1, 1, 1]], [0, 1], [1, -1])


@pytest.mark.flaky(reruns=3)
def test_configure_architecture(make_model):
    model = make_model(layers=[dict(activation='relu', size=128), dict(activation='relu', size=32)])

    train(model, episodes=200)

    assert model.estimate([[0, 0, 0]]).squeeze()[0] == approx(1, abs=0.001)
    assert model.estimate([[1, 1, 1]]).squeeze()[1] == approx(-1, abs=0.001)


@pytest.mark.flaky(reruns=3)
def test_linear_interpolation_between_models(make_model):
    base = make_model()
    target = make_model()

    train(target, episodes=500)

    base.linear_interpolate(target, 0.1)

    assert base.estimate([[0, 0, 0]]).squeeze()[0] != approx(1, abs=0.3)
    assert base.estimate([[1, 1, 1]]).squeeze()[1] != approx(-1, abs=0.3)

    base.linear_interpolate(target, 0.9)

    assert base.estimate([[0, 0, 0]]).squeeze()[0] == approx(1, abs=0.3)
    assert base.estimate([[1, 1, 1]]).squeeze()[1] == approx(-1, abs=0.3)
