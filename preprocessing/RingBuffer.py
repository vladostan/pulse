import numpy as np
from collections import Sequence



class RingBuffer(Sequence):
    def __init__(self, capacity, dtype=float, allow_overwrite=True):
        self._arr = np.zeros(capacity, dtype)
        self._left_index = 0
        self._right_index = 0
        self._capacity = capacity
        self._allow_overwrite = allow_overwrite

    def _unwrap(self):
        return np.concatenate((
            self._arr[self._left_index:min(self._right_index, self._capacity)],
            self._arr[:max(self._right_index - self._capacity, 0)]
        ))

    def _fix_indices(self):
        if self._left_index >= self._capacity:
            self._left_index -= self._capacity
            self._right_index -= self._capacity
        elif self._left_index < 0:
            self._left_index += self._capacity
            self._right_index += self._capacity

    @property
    def is_full(self):
        return len(self) == self._capacity

    # numpy compatibility
    def __array__(self):
        return self._unwrap()

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def shape(self):
        return (len(self),) + self._arr.shape[1:]

    # these mirror methods from deque
    @property
    def maxlen(self):
        return self._capacity

    def append(self, value):
        if self.is_full:
            if not self._allow_overwrite:
                raise IndexError('append to a full RingBuffer with overwrite disabled')
            elif not len(self):
                return
            else:
                self._left_index += 1
        self._arr[self._right_index % self._capacity] = value
        self._right_index += 1
        self._fix_indices()

    def pop(self):
        if len(self) == 0:
            raise IndexError("pop from an empty RingBuffer")
        self._right_index -= 1
        self._fix_indices()
        res = self._arr[self._right_index % self._capacity]
        return res

    def pop_window(self, window_size, overlap):
        dataset = []
        for i in range(0, window_size):
            dataset.append(self.pop())
        self._right_index = self._right_index + overlap
        self._fix_indices()
        return dataset

    # implement Sequence methods
    def __len__(self):
        return self._right_index - self._left_index

    def __getitem__(self, item):
        return self._unwrap()[item]

    def __iter__(self):
        return iter(self._unwrap())

    # Everything else
    def __repr__(self):
        return '<RingBuffer of {!r}>'.format(np.asarray(self))


# buffer = RingBuffer(100)
# buffer.append(78)
# print buffer.pop_window(1)