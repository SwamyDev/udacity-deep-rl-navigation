import pytest
import torch
import numpy as np
from pytest import approx

from udacity_rl.model import QModel, Actor, Critic
from torch.nn import functional as F


@pytest.fixture
def make_q_model():
    def factory(obs_size=3, act_size=2, layers=None):
        layers = layers or [dict(activation='relu', size=64)]
        return QModel(obs_size, act_size, layers)

    return factory


@pytest.fixture
def q_model(make_q_model):
    return make_q_model()


@pytest.fixture
def make_actor():
    def factory(obs_size=3, act_size=2, layers=None):
        layers = layers or [dict(activation='relu', size=64)]
        return Actor(obs_size, act_size, layers)

    return factory


@pytest.fixture
def actor(make_actor):
    return make_actor()


@pytest.fixture
def make_critic():
    def factory(obs_size=3, act_size=2, layers=None):
        layers = layers or [dict(activation='relu', size=64), dict(activation='relu', size=64)]
        return Critic(obs_size, act_size, layers)

    return factory


@pytest.fixture
def critic(make_critic):
    return make_critic()


def test_estimates_have_expected_shape(q_model):
    assert q_model.estimate([[0, 0, 0]]).shape == (1, 2)


@pytest.mark.flaky(reruns=3)
def test_fitting_observations_to_target_values(q_model):
    assert q_model.estimate([[0, 0, 0]]).squeeze()[0] != approx(1, abs=0.001)
    assert q_model.estimate([[1, 1, 1]]).squeeze()[1] != approx(-1, abs=0.001)

    train_q(q_model, episodes=1000)

    assert q_model.estimate([[0, 0, 0]]).squeeze()[0] == approx(1, abs=0.001)
    assert q_model.estimate([[1, 1, 1]]).squeeze()[1] == approx(-1, abs=0.001)


def train_q(model, episodes):
    for _ in range(episodes):
        model.fit([[0, 0, 0], [1, 1, 1]], [0, 1], [1, -1])


@pytest.mark.flaky(reruns=3)
def test_configure_architecture(make_q_model):
    model = make_q_model(layers=[dict(activation='relu', size=128), dict(activation='relu', size=32)])

    train_q(model, episodes=200)

    assert model.estimate([[0, 0, 0]]).squeeze()[0] == approx(1, abs=0.001)
    assert model.estimate([[1, 1, 1]]).squeeze()[1] == approx(-1, abs=0.001)


@pytest.mark.flaky(reruns=3)
def test_linear_interpolation_between_models(make_q_model):
    base = make_q_model()
    target = make_q_model()

    train_q(target, episodes=500)

    base.linear_interpolate(target, 0.1)

    assert base.estimate([[0, 0, 0]]).squeeze()[0] != approx(1, abs=0.3)
    assert base.estimate([[1, 1, 1]]).squeeze()[1] != approx(-1, abs=0.3)

    base.linear_interpolate(target, 0.9)

    assert base.estimate([[0, 0, 0]]).squeeze()[0] == approx(1, abs=0.3)
    assert base.estimate([[1, 1, 1]]).squeeze()[1] == approx(-1, abs=0.3)


@pytest.mark.flaky(reruns=3)
def test_save_and_load(make_q_model, tmp_path):
    trained = make_q_model()
    train_q(trained, episodes=500)

    assert trained.estimate([[0, 0, 0]]).squeeze()[0] == approx(1, abs=0.3)
    assert trained.estimate([[1, 1, 1]]).squeeze()[1] == approx(-1, abs=0.3)

    trained.save(tmp_path / 'checkpoint.pth')

    loaded = make_q_model()

    assert loaded.estimate([[0, 0, 0]]).squeeze()[0] != approx(1, abs=0.3)
    assert loaded.estimate([[1, 1, 1]]).squeeze()[1] != approx(-1, abs=0.3)

    loaded.load(tmp_path / 'checkpoint.pth')

    assert loaded.estimate([[0, 0, 0]]).squeeze()[0] == approx(1, abs=0.3)
    assert loaded.estimate([[1, 1, 1]]).squeeze()[1] == approx(-1, abs=0.3)


def test_actor_estimates_have_expected_shape(actor):
    assert actor.estimate([[0, 0, 0]]).shape == (1, 2)


def test_actor_minimize_loss(actor):
    obs = torch.from_numpy(np.array([[0, 0, 0]], dtype=np.float32))
    target = torch.from_numpy(np.array([[1, 1]], dtype=np.float32))
    for _ in range(1000):
        act = actor(obs)
        loss = -(act - target).mean()
        actor.minimize(loss)

    assert_actor_estimates_eq(actor.estimate(obs), np.array([[1, 1]]))


def assert_actor_estimates_eq(actual, expected):
    np.testing.assert_array_almost_equal(actual, expected, decimal=2)


def test_critic_returns_scalar_value(critic):
    obs = torch.from_numpy(np.array([[0, 0, 0]], dtype=np.float32))
    act = torch.from_numpy(np.array([[0, 0]], dtype=np.float32))
    assert critic(obs, act).cpu().detach().numpy().shape == (1, 1)


def test_critic_fit_q_target(critic):
    obs = torch.from_numpy(np.array([[0, 1, 0]], dtype=np.float32))
    act = torch.from_numpy(np.array([[1, -1]], dtype=np.float32))
    target = torch.from_numpy(np.array([[10]], dtype=np.float32))
    for _ in range(1000):
        exp = critic(obs, act)
        loss = F.mse_loss(exp, target)
        critic.minimize(loss)

    assert_critic_val_eq(critic(obs, act).cpu().detach().numpy(), np.array([[10]]))


def assert_critic_val_eq(actual, expected):
    np.testing.assert_array_almost_equal(actual, expected, decimal=5)
