from collections import Counter

import numpy as np
import pytest

from udacity_rl.memory import Memory, MemoryRecordError, UniformReplayBuffer, PrioritizedReplayBuffer, calc_priority


@pytest.fixture
def make_memory():
    def factory(batch_size=64, record_size=10000, num_records=0, seed=None):
        m = Memory(batch_size, UniformReplayBuffer(record_size, seed))
        for index in range(num_records):
            m.record(index=index)
        return m

    return factory


@pytest.fixture
def make_priority():
    def factory(batch_size=64, record_size=10000, num_records=0, td_error=None, alpha=1, seed=None):
        if td_error is None:
            td_error = [1] * num_records
        priorities = [calc_priority(td_e, alpha) for td_e in td_error]

        m = Memory(batch_size, PrioritizedReplayBuffer(record_size, seed))
        for index, p in enumerate(priorities):
            m.record(index=index, priority=p)

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
    memory.record(index=0)
    assert len(memory) == 1
    memory.record(index=1)
    assert len(memory) == 2


@pytest.mark.parametrize('batch_size', (1, 3))
def test_sampling_memory_returns_list_of_experiences_if_enough_records_to_fill_a_batch(make_memory, batch_size):
    memory = make_memory(batch_size, num_records=batch_size)
    assert sorted(memory.sample()) == sorted([np.array([index]) for index in range(batch_size)])


def test_sampling_memory_returns_empty_list_if_not_enough_records_to_fill_a_batch(make_memory):
    assert make_memory(batch_size=2, num_records=1).sample() == []


def test_is_unfilled_indicated_whether_batch_is_incomplete_or_not(make_memory):
    assert make_memory(batch_size=2, num_records=1).is_unfilled()
    assert not make_memory(batch_size=2, num_records=2).is_unfilled()


def test_sample_randomly_from_record_if_record_exceeds_batch_size(make_memory):
    assert (make_memory(batch_size=2, num_records=10, seed=17).sample() !=
            make_memory(batch_size=2, num_records=10, seed=42).sample()).any()


def test_record_has_fixed_length(make_memory):
    assert len(make_memory(batch_size=1, record_size=3, num_records=10)) == 3


def test_recording_multiple_attributes(make_memory):
    memory = make_memory(batch_size=2, seed=42)
    memory.record(index=0, state_value=[0, 1], action_value=1.4)
    memory.record(index=1, state_value=[1, 0], action_value=-0.2)
    assert_samples_eq(memory.sample(), (np.array([0, 1]), np.array([[0, 1], [1, 0]]), np.array([1.4, -0.2])))


def assert_samples_eq(actual, expected):
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)


def test_multiple_attributes_keep_association_when_randomly_sampled(make_memory):
    memory = make_memory(batch_size=2, seed=42)
    memory.record(index=0, state_value=[0, 1], action_value=1.4)
    memory.record(index=1, state_value=[1, 0], action_value=-0.2)
    memory.record(index=2, state_value=[1, 1], action_value=-1.2)
    memory.record(index=3, state_value=[0, 0], action_value=2.1)
    assert_samples_eq(memory.sample(), (np.array([0, 3]), np.array([[0, 1], [0, 0]]), np.array([1.4, 2.1])))


def test_recording_different_attributes_raises_error(memory):
    memory.record(index=0, state_value=[0, 1])
    with pytest.raises(MemoryRecordError):
        memory.record(index=1)
    with pytest.raises(MemoryRecordError):
        memory.record(index=1, staet_value=[0, 1])
    with pytest.raises(MemoryRecordError):
        memory.record(state_value=[0, 1], index=1)


def test_sampling_memory_returns_numpy_arrays(make_memory):
    memory = make_memory(batch_size=2, num_records=2)
    sample = memory.sample()
    assert isinstance(sample, np.ndarray)


def test_multiple_attributes_are_als_returned_as_numpy_arrays(make_memory):
    memory = make_memory(batch_size=2)
    memory.record(index=0, state_value=[0, 1])
    memory.record(index=1, state_value=[1, 1])
    index, state_value = memory.sample()
    assert isinstance(index, np.ndarray) and isinstance(state_value, np.ndarray)


def test_sampling_from_priority_replay_buffer_initially_is_uniformly_distributed(make_priority):
    memory = make_priority(batch_size=3, num_records=10)
    actual_dist = determine_memory_distribution(memory)
    assert_distribution(actual_dist, np.repeat(1 / 10, repeats=10))


def determine_memory_distribution(memory):
    c = Counter()
    for _ in range(10000):
        index, leafs, weights = memory.sample()
        for idx in index:
            c[idx] += 1

    return np.array([v for _, v in c.most_common()]) / sum(c.values())


def assert_distribution(actual, expected):
    np.testing.assert_array_almost_equal(actual, expected, decimal=2)


def test_priority_replay_buffer_values_integrate_into_return_tuple(make_priority):
    memory = make_priority(batch_size=2)
    memory.record(index=0, state_value=[0, 1], td_error=1)
    memory.record(index=1, state_value=[1, 1], td_error=1)
    index, state_value, leafs, weights = memory.sample()
    assert isinstance(index, np.ndarray) and isinstance(state_value, np.ndarray) and \
           isinstance(leafs, np.ndarray) and isinstance(weights, np.ndarray)


def test_prioritized_replay_is_proportional_to_updated_values(make_priority):
    td_errors = [10, 9, 8, 6, 5, 4, 3, 2, 1, 1]
    memory = make_priority(batch_size=3, num_records=10, td_error=td_errors)
    actual_dist = determine_memory_distribution(memory)
    assert_distribution(actual_dist, np.array(td_errors) / sum(td_errors))


def test_updating_leaf_changes_priority(make_priority):
    memory = make_priority(batch_size=1, num_records=3, td_error=[1] * 3)
    _, leafs, _ = memory.sample()
    leafs[0].update(100)
    actual_dist = determine_memory_distribution(memory)
    assert_distribution(actual_dist, np.array([100, 1, 1]) / 102)


def test_alpha_value_controls_priority_behaviour(make_priority):
    memory = make_priority(batch_size=1, num_records=3, td_error=[1, 100, 10], alpha=0)
    assert_distribution(determine_memory_distribution(memory), np.repeat(1 / 3, repeats=3))

    memory = make_priority(batch_size=1, num_records=3, td_error=[1, 100, 10], alpha=0.5)
    assert_distribution(determine_memory_distribution(memory), np.array([0.7052, 0.2203, 0.0745]))


def test_priority_replay_buffer_calculates_corrective_weight(make_priority):
    memory = make_priority(batch_size=3, num_records=3, td_error=[30, 50, 20])
    weights = get_full_sample_weights(memory, beta=1)
    assert_weights(weights, np.array([0.4, 2/3, 1.0]))
    weights = get_full_sample_weights(memory, beta=0)
    assert_weights(weights, np.array([1.0, 1.0, 1.0]))
    weights = get_full_sample_weights(memory, beta=0.5)
    assert_weights(weights, np.array([0.633, 0.817, 1.0]))


def get_full_sample_weights(memory, beta):
    weights = []
    while len(weights) == 0:
        index, _, ws = memory.sample(beta=beta)
        if 0 in index and 1 in index and 2 in index:
            weights = ws
    return sorted(weights)


def assert_weights(actual, expected):
    np.testing.assert_array_almost_equal(actual, expected, decimal=3)
