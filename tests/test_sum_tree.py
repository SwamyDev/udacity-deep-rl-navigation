import pytest

from udacity_rl.sum_tree import SumTree


@pytest.fixture
def tree():
    return SumTree(capacity=3)


def test_is_initially_empty(tree):
    assert len(tree) == 0
    assert tree.total == 0


def test_tree_with_a_single_value_has_correct_size_and_total(tree):
    tree.add(10, "A")
    assert len(tree) == 1
    assert tree.total == 10


def test_tree_with_n_values_has_correct_size_and_total(tree):
    tree.add(10, "A")
    tree.add(5, "B")
    tree.add(12, "C")
    assert len(tree) == 3
    assert tree.total == 27

def test_tree_queries_first_segment(tree):
    tree.add(10, "A")
    assert_node(tree.query(0), 10, "A")


def assert_node(node, exp_val, exp_data):
    assert node.value == exp_val and node.data == exp_data


def test_tree_queries_n_segments(tree):
    tree.add(10, "A")
    tree.add(5, "B")
    assert_node(tree.query(0.0), 10, "A")
    assert_node(tree.query(10), 5, "B")
    tree.add(12, "C")
    assert_node(tree.query(0.0), 10, "A")
    assert_node(tree.query(10), 5, "B")
    assert_node(tree.query(15), 12, "C")


def test_tree_reaches_maximum_capacity(tree):
    tree.add(10, "A")
    tree.add(5, "B")
    tree.add(12, "C")
    tree.add(14, "D")
    assert len(tree) == 3
    assert tree.total == 31
    assert_node(tree.query(0.0), 14, "D")
    assert_node(tree.query(14.0), 5, "B")
    assert_node(tree.query(19.0), 12, "C")


def test_return_right_most_value_when_overshooting(tree):
    tree.add(10, "A")
    tree.add(5, "B")
    assert_node(tree.query(15.1), 5, "B")
    tree.add(12, "C")
    assert_node(tree.query(27.1), 12, "C")
    tree.add(15, "D")
    assert_node(tree.query(42.1), 15, "D")


def test_updating_leaf_without_data_just_updates_value(tree):
    tree.add(10, "A")
    tree.add(5, "B")
    tree.query(0.0).update(7.0)
    assert_node(tree.query(6.9), 7, "A")
    assert_node(tree.query(7.0), 5, "B")
    assert tree.total == 12
