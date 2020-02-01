from collections import deque
from contextlib import contextmanager

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytest import approx


def _identity(x):
    return x


class FeedForwardNetwork(nn.Module):
    _ACTIVATION_MAPPING = {
        'relu': F.relu,
        'identity': _identity,
    }

    def __init__(self, input_size, layers, seed):
        super(FeedForwardNetwork, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self._layers = self._create_layers(input_size, layers)

    def _create_layers(self, in_size, layer_cfg):
        layers = list()
        layer_cfg = deque(layer_cfg)
        while len(layer_cfg):
            cfg = layer_cfg.popleft()
            activation_fn, size = self._ACTIVATION_MAPPING[cfg['activation']], cfg['size']
            fc_name = f"fc{len(layers)}"
            setattr(self, fc_name, nn.Linear(in_size, size))
            layers.append((activation_fn, getattr(self, fc_name)))
            in_size = size

        return layers

    def forward(self, state):
        for activation, linear in self._layers:
            state = activation(linear(state))
        return state


@contextmanager
def _eval_scope(ann):
    ann.eval()
    try:
        yield
    finally:
        ann.train()


class QModel:
    def __init__(self, observation_size, action_size, layers, lr=5e-4, device='cpu', seed=None):
        layers.append(dict(activation='identity', size=action_size))
        self._ann = FeedForwardNetwork(observation_size, layers, seed)
        self._optimizer = optim.Adam(self._ann.parameters(), lr=lr)
        self._device = device

    def estimate(self, observations):
        obs = torch.from_numpy(np.array(observations, dtype=np.float32)).to(self._device)
        with torch.no_grad(), _eval_scope(self._ann):
            qs = self._ann(obs).detach().numpy()
        return qs

    def fit(self, observations, actions, targets):
        obs = torch.from_numpy(np.array(observations, dtype=np.float32)).to(self._device)
        acts = torch.from_numpy(np.array(actions, dtype=np.int)).to(self._device)
        est = self._ann(obs)
        current = est.gather(1, acts)

        believe = torch.from_numpy(np.array(targets, dtype=np.float32)).to(self._device)
        loss = F.mse_loss(current, believe)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def linear_interpolate(self, target, tau):
        for mine, target in zip(self.parameters(), target.parameters()):
            mine.data.copy_((1 - tau) * mine.data + tau * target.data)

    def parameters(self):
        return self._ann.parameters()


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


def test_fitting_observations_to_target_values(model):
    assert model.estimate([[0, 0, 0]]).squeeze()[0] != approx(1, abs=0.001)
    assert model.estimate([[1, 1, 1]]).squeeze()[1] != approx(-1, abs=0.001)

    for _ in range(1000):
        model.fit([[0, 0, 0], [1, 1, 1]], [[0], [1]], [[1], [-1]])

    assert model.estimate([[0, 0, 0]]).squeeze()[0] == approx(1, abs=0.001)
    assert model.estimate([[1, 1, 1]]).squeeze()[1] == approx(-1, abs=0.001)


def test_configure_architecture(make_model):
    model = make_model(layers=[dict(activation='relu', size=128), dict(activation='relu', size=32)])

    for _ in range(200):
        model.fit([[0, 0, 0], [1, 1, 1]], [[0], [1]], [[1], [-1]])

    assert model.estimate([[0, 0, 0]]).squeeze()[0] == approx(1, abs=0.001)
    assert model.estimate([[1, 1, 1]]).squeeze()[1] == approx(-1, abs=0.001)


def test_linear_interpolation_between_models(make_model):
    base = make_model()
    target = make_model()
    for _ in range(500):
        target.fit([[0, 0, 0], [1, 1, 1]], [[0], [1]], [[1], [-1]])

    base.linear_interpolate(target, 0.1)

    assert base.estimate([[0, 0, 0]]).squeeze()[0] != approx(1, abs=0.3)
    assert base.estimate([[1, 1, 1]]).squeeze()[1] != approx(-1, abs=0.3)

    base.linear_interpolate(target, 0.9)

    assert base.estimate([[0, 0, 0]]).squeeze()[0] == approx(1, abs=0.3)
    assert base.estimate([[1, 1, 1]]).squeeze()[1] == approx(-1, abs=0.3)
