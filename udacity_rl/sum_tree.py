from itertools import zip_longest


class SumNode:
    def __init__(self, left, right):
        self.value = 0
        self.data = None
        self.parent = None
        self.left = left
        self.right = right
        self._update_parent()

    def _update_parent(self):
        if self.left:
            self.left.parent = self
        if self.right:
            self.right.parent = self

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    def propagate(self, change):
        self.value += change
        if self.parent:
            self.parent.propagate(change)

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return f"Node({self.value}, {self.data})"

    def query(self, segment):
        if segment < self.left.value:
            return self.left.query(segment)
        else:
            return self.right.query(segment - self.left.value)


class SumLeaf(SumNode):
    def __init__(self, value, data):
        super().__init__(None, None)
        self.value = value
        self.data = data

    def query(self, segment):
        return self

    def update(self, new_value, data):
        change = new_value - self.value
        self.value = new_value
        self.data = data
        if self.parent:
            self.parent.propagate(change)


class SumTree:
    Node = SumNode
    Leaf = SumLeaf

    def __init__(self, capacity):
        self._capacity = capacity
        self._size = 0
        self._leafs = [self.Leaf(0, None) for _ in range(self._capacity)]
        self._root = self._make_empty_tree()
        self._cursor = 0

    def _make_empty_tree(self):
        nodes = self._leafs
        while len(nodes) != 1:
            it_nds = iter(nodes)
            nodes = [self.Node(*pair) for pair in zip_longest(it_nds, it_nds)]
        return nodes[0]

    def __len__(self):
        return self._size

    @property
    def total(self):
        return self._root.value

    def query(self, segment):
        if segment >= self.total:
            raise self.SegmentOutOfBoundsError(f"The requested segment {segment} is out of bounds ({self.total})")

        return self._root.query(segment)

    def add(self, value, data):
        self._leafs[self._cursor].update(value, data)
        self._cursor = (self._cursor + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    class SegmentOutOfBoundsError(IndexError):
        pass
