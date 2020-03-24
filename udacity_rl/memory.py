import abc
import logging
import random
from collections import deque

import numpy as np

from udacity_rl.sum_tree import SumTree

logger = logging.getLogger(__name__)


class ReplayBuffer(abc.ABC):
    @abc.abstractmethod
    def append(self, data):
        pass

    @abc.abstractmethod
    def sample(self, size, **kwargs):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


class UniformReplayBuffer(ReplayBuffer):
    def __init__(self, record_size, seed=None):
        self._record = deque(maxlen=record_size)
        if seed is not None:
            random.seed(seed)

        self._print_config()

    def _print_config(self):
        logger.info(f"Uniform Replay Buffer:\n"
                    f"\tRecord size:\t{self._record.maxlen}\n")

    def append(self, data):
        self._record.append(data)

    def sample(self, size, _=None):
        return random.sample(self._record, k=size)

    def __len__(self):
        return len(self._record)


def calc_priority(td_error, alpha, epsilon=0.1):
    return (np.abs(td_error) + epsilon) ** alpha


class PrioritizedReplayBuffer(ReplayBuffer):
    PRIORITY_INDEX = -1

    def __init__(self, record_size, seed=None):
        self._sum_tree = SumTree(record_size)
        if seed is not None:
            random.seed(seed)

        self._print_config()

    def _print_config(self):
        logger.info(f"Prioritized Replay Buffer:\n"
                    f"\tRecord size:\t{self._sum_tree.capacity}\n")

    def append(self, data):
        self._sum_tree.add(data[self.PRIORITY_INDEX], data[:self.PRIORITY_INDEX])

    def sample(self, size, beta=0.5):
        batch = []
        weights = []
        sum_p = self._sum_tree.total
        n_tree = len(self._sum_tree)
        max_w = 0

        segment = sum_p / size
        for i in range(size):
            s = random.uniform(segment * i, segment * (i + 1))
            leaf = self._sum_tree.query(s)

            w = np.power(n_tree * (leaf.value / sum_p), -beta)
            if w > max_w:
                max_w = w

            weights.append(w)
            batch.append(leaf.data + (leaf,))

        for i in range(size):
            batch[i] += tuple([weights[i] / max_w])

        return batch

    def __len__(self):
        return len(self._sum_tree)


class Memory:
    def __init__(self, batch_size, buffer):
        self._buffer = buffer
        self._batch_size = batch_size
        self._keys = None
        self._print_config()

    def _print_config(self):
        logger.info(f"Memory configuration:\n"
                    f"\tBatch size:\t{self._batch_size}\n")

    def record(self, **kwargs):
        keys = tuple(kwargs.keys())
        if self._keys is None:
            self._keys = keys
        if keys != self._keys:
            raise MemoryRecordError(f"The recorded value keys are not allowed:\n"
                                    f"expected: {self._keys}\nactual:{keys}")
        self._buffer.append(tuple(kwargs[k] for k in kwargs))

    def sample(self, **kwargs):
        if self.is_unfilled():
            return []

        sample = self._buffer.sample(self._batch_size, **kwargs)
        if len(sample[0]) > 1:
            return self._cast_to_ndarray_tuple(list(zip(*sample)))

        return np.array(sample)

    @staticmethod
    def _cast_to_ndarray_tuple(attributes):
        for i in range(len(attributes)):
            attributes[i] = np.array(attributes[i])
        return tuple(attributes)

    def is_unfilled(self):
        return len(self) < self._batch_size

    def __len__(self):
        return len(self._buffer)


class MemoryRecordError(AssertionError):
    pass
