import random

import pytest


class Memory:
    def __init__(self, batch_size, seed=None):
        self._record = list()
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


class ExperienceStub:
    def __init__(self, index):
        self.index = index

    def __eq__(self, other):
        return self.index == other.index

    def __lt__(self, other):
        return self.index < other.index

    def __repr__(self):
        return f"ExperienceStub({self.index})"


@pytest.fixture
def make_memory():
    def factory(batch_size=64, record_size=0, seed=None):
        m = Memory(batch_size, seed)
        for index in range(record_size):
            m.record(ExperienceStub(index))
        return m

    return factory


@pytest.fixture
def memory(make_memory):
    return make_memory()


def test_memory_is_initially_empty(memory):
    assert len(memory) == 0


def test_sampling_empty_memory_returns_empty_list(memory):
    assert memory.sample() == []


def test_recording_experience_increases_length(memory):
    memory.record(ExperienceStub(0))
    assert len(memory) == 1
    memory.record(ExperienceStub(1))
    assert len(memory) == 2


@pytest.mark.parametrize('batch_size', (1, 3))
def test_sampling_memory_returns_list_of_experiences_if_enough_records_to_fill_a_batch(make_memory, batch_size):
    memory = make_memory(batch_size, record_size=batch_size)
    assert sorted(memory.sample()) == sorted([ExperienceStub(index) for index in range(batch_size)])


def test_sampling_memory_returns_empty_list_if_not_enough_records_to_fill_a_batch(make_memory):
    memory = make_memory(batch_size=2, record_size=1)
    assert memory.sample() == []


def test_sample_randomly_from_record_if_record_exceeds_batch_size(make_memory):
    assert make_memory(batch_size=2, record_size=10, seed=17).sample() != \
           make_memory(batch_size=2, record_size=10, seed=42).sample()
