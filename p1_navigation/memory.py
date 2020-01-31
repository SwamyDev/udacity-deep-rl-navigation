import random
from collections import deque


class Memory:
    def __init__(self, batch_size, record_size, seed=None):
        self._record = deque(maxlen=record_size)
        self._batch_size = batch_size
        if seed is not None:
            random.seed(seed)

    def record(self, experience):
        self._record.append(experience)

    def sample(self):
        if len(self) < self._batch_size:
            return []

        return random.sample(self._record, k=self._batch_size)

    def __len__(self):
        return len(self._record)
