import abc
import logging
import random
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class ReplayBuffer(abc.ABC):
    @abc.abstractmethod
    def append(self, data):
        pass

    @abc.abstractmethod
    def sample(self, size):
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

    def sample(self, size):
        return random.sample(self._record, k=size)

    def __len__(self):
        return len(self._record)


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

    def sample(self):
        if self.is_unfilled():
            return []

        sample = self._buffer.sample(self._batch_size)
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
