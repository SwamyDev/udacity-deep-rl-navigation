import pytest

from p1_navigation.memory import Memory


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
    def factory(batch_size=64, record_size=10000, num_records=0, seed=None):
        m = Memory(batch_size, record_size, seed)
        for index in range(num_records):
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
    memory = make_memory(batch_size, num_records=batch_size)
    assert sorted(memory.sample()) == sorted([ExperienceStub(index) for index in range(batch_size)])


def test_sampling_memory_returns_empty_list_if_not_enough_records_to_fill_a_batch(make_memory):
    memory = make_memory(batch_size=2, num_records=1)
    assert memory.sample() == []


def test_sample_randomly_from_record_if_record_exceeds_batch_size(make_memory):
    assert make_memory(batch_size=2, num_records=10, seed=17).sample() != \
           make_memory(batch_size=2, num_records=10, seed=42).sample()


def test_record_has_fixed_length(make_memory):
    assert len(make_memory(batch_size=1, record_size=3, num_records=10)) == 3
