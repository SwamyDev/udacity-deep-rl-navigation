import random
from collections import deque


class Memory:
    def __init__(self, batch_size, record_size, seed=None):
        self._record = deque(maxlen=record_size)
        self._batch_size = batch_size
        self._keys = None
        if seed is not None:
            random.seed(seed)

    def record(self, **kwargs):
        keys = tuple(kwargs.keys())
        if self._keys is None:
            self._keys = keys
        if keys != self._keys:
            raise MemoryRecordError(f"The recorded value keys are not allowed:\n"
                                    f"expected: {self._keys}\nactual:{keys}")
        self._record.append(tuple(kwargs[k] for k in kwargs))

    def sample(self):
        if self.is_unfilled():
            return []

        sample = random.sample(self._record, k=self._batch_size)
        if len(sample[0]) > 1:
            return tuple(zip(*sample))
        return sample

    def is_unfilled(self):
        return len(self) < self._batch_size

    def __len__(self):
        return len(self._record)


class MemoryRecordError(AssertionError):
    pass
