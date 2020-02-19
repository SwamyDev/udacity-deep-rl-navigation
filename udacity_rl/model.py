from collections import deque
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn as nn, optim as optim
from torch.nn import functional as F


def _identity(x):
    return x


class ConfigurableNetwork(nn.Module):
    _ACTIVATION_MAPPING = {
        'relu': F.relu,
        'tanh': F.tanh,
        'identity': _identity,
    }

    def __init__(self, input_size, layers, seed):
        super(ConfigurableNetwork, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self._layers = self._create_layers(input_size, layers)
        self._print_architecture()

    def _print_architecture(self):
        print("FeedForwardNetwork architecture:")
        for activation, linear in self._layers:
            print(f"\t{activation.__name__}({linear.in_features}x{linear.out_features})")
        print("\n")

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

    def _forward_through_layers(self, state):
        for activation, linear in self._layers:
            state = activation(linear(state))
        return state


class FeedForwardNetwork(ConfigurableNetwork):
    def forward(self, state):
        return self._forward_through_layers(state)


class CriticNetwork(ConfigurableNetwork):
    def __init__(self, observation_size, action_size, layers, seed):
        head_cfg = layers[0]
        super().__init__(head_cfg['size'] + action_size, layers[1:], seed)
        self._head_fc = nn.Linear(observation_size, head_cfg['size'])
        self._head_fn = self._ACTIVATION_MAPPING[head_cfg['activation']]
        self._tail_fc = nn.Linear(layers[-1]['size'], 1)

    def forward(self, state, action):
        x = self._head_fn(self._head_fc(state))
        x = self._forward_through_layers(torch.cat((x, action), dim=1))
        return self._tail_fc(x)


@contextmanager
def _eval_scope(ann):
    ann.eval()
    try:
        yield
    finally:
        ann.train()


class Model:
    def __init__(self, input_size, layers, lr=5e-4, device='cpu', seed=None):
        self._device = device
        self._input_size = input_size
        self._ann = self._make_ann(layers, seed)
        self._optimizer = optim.Adam(self._ann.parameters(), lr=lr)

        print(f"Qodel configuration:\n"
              f"\tLearning rate:\t{lr}\n"
              f"\tDevice:\t{device}\n")

    def _make_ann(self, layers, seed):
        return FeedForwardNetwork(self._input_size, layers, seed).to(self._device)

    def minimize(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def linear_interpolate(self, target, tau):
        for mine, target in zip(self.parameters(), target.parameters()):
            mine.data.copy_((1 - tau) * mine.data + tau * target.data)

    def parameters(self):
        return self._ann.parameters()

    def save(self, file):
        torch.save(self._ann.state_dict(), file)

    def load(self, file):
        self._ann.load_state_dict(torch.load(file))


class Estimator(Model):
    def estimate(self, observations):
        obs = torch.from_numpy(np.array(observations, dtype=np.float32)).to(self._device)
        with torch.no_grad(), _eval_scope(self._ann):
            qs = self._ann(obs).cpu().detach().numpy()
        return qs


class QModel(Estimator):
    def __init__(self, observation_size, action_size, layers, lr=5e-4, device='cpu', seed=None):
        layers = list(layers)
        layers.append(dict(activation='identity', size=action_size))
        super().__init__(observation_size, layers, lr, device, seed)

    def fit(self, observations, actions, targets):
        obs = torch.from_numpy(np.vstack(observations)).float().to(self._device)
        acts = torch.from_numpy(np.vstack(actions)).long().to(self._device)
        est = self._ann(obs)
        current = est.gather(1, acts)

        believe = torch.from_numpy(np.vstack(targets)).float().to(self._device)
        loss = F.mse_loss(current, believe)
        self.minimize(loss)


class Actor(Estimator):
    def __init__(self, observation_size, action_size, layers, lr=5e-4, device='cpu', seed=None):
        layers = list(layers)
        layers.append(dict(activation='tanh', size=action_size))
        super().__init__(observation_size, layers, lr, device, seed)

    def __call__(self, observation):
        return self._ann(observation)


class Critic(Model):
    def __init__(self, observation_size, action_size, layers, lr=5e-4, device='cpu', seed=None):
        self._observation_size = observation_size
        self._action_size = action_size
        super().__init__(observation_size, layers, lr, device, seed)

    def _make_ann(self, layers, seed):
        return CriticNetwork(self._observation_size, self._action_size, layers, seed).to(self._device)

    def __call__(self, observations, actions):
        return self._ann(observations, actions)
