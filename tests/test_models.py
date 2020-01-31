from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytest import approx


class FeedForwardNetwork(nn.Module):
    def __init__(self, observation_size, action_size, seed):
        super(FeedForwardNetwork, self).__init__()
        self._hidden = nn.Linear(observation_size, 64)
        self._output = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self._hidden(state))
        return self._output(x)


@contextmanager
def _in_estimation(ann):
    ann.eval()
    try:
        yield
    finally:
        ann.train()


class QModel:
    def __init__(self, observation_size, action_size, lr=5e-4, device='cpu', seed=None):
        self._ann = FeedForwardNetwork(observation_size, action_size, seed)
        self._optimizer = optim.Adam(self._ann.parameters(), lr=lr)
        self._device = device

    def estimate(self, observations):
        obs = torch.from_numpy(np.array(observations, dtype=np.float32)).to(self._device)
        with torch.no_grad(), _in_estimation(self._ann):
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


def test_estimates_have_expected_shape():
    model = QModel(3, 2)
    assert model.estimate([[0, 0, 0]]).shape == (1, 2)


def test_fitting_observations_to_target_values():
    model = QModel(3, 2)

    assert model.estimate([[0, 0, 0]]).squeeze()[0] != approx(1)
    assert model.estimate([[1, 1, 1]]).squeeze()[1] != approx(-1)

    for _ in range(1000):
        model.fit([[0, 0, 0], [1, 1, 1]], [[0], [1]], [[1], [-1]])

    assert model.estimate([[0, 0, 0]]).squeeze()[0] == approx(1)
    assert model.estimate([[1, 1, 1]]).squeeze()[1] == approx(-1)
