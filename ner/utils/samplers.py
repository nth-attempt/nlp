from typing import Any, Callable, List, Union, Tuple
from heapq import heappop, heappush

import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler


class BucketBatchSampler(Sampler):
    # https://github.com/pytorch/text/pull/859/files
    """Defines a batch sampler that batches examples of similar lengths together and
    minimizes amount of padding needed. This BatchSampler works by categorizing each
    raw data by putting them in a bucket whose lengths are in the upperbound range of
    ``bucket_boundaries``. For ``bucket_boundaries`` = [5, 10], there will be three
    different buckets that will consist of sentences whose lengths are less than 5,
    between 5 and 10, and more than 10.
    Arguments:
        data_source: data source to sample from.
        bucket_boundaries: upper length boundaries to merge sentences with length
            less than or equal to the boundaries.
        seq_len_fn: function to return the current length of the sequence.
        batch_size: size of mini-batch.
            Default: 32
        shuffle: data_source will be wrapped with RandomSampler if set to ``True``,
            otherwise, SequentialSampler. Default: True
    Example:
        >>> dummy = [
            torch.tensor(range(1, torch.randint(2, 11, (1,))[0])) for num in range(10)
        ]
        >>> def tensor_seq_len_fn(row):
        ...     return row.size(0)
        >>> list(BucketBatchSampler(dummy, [5, 10], tensor_seq_len_fn, batch_size=5, shuffle=False))
        [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9]]
        >>> list(BucketBatchSampler(dummy, [5, 10], tensor_seq_len_fn, batch_size=5))
        [[9, 2, 4, 3, 1], [8, 7, 5, 6], [0]]
    """

    def __init__(
        self,
        data_source: Dataset,
        bucket_boundaries: List[int],
        seq_len_fn: Callable[[Union[List[Any], torch.Tensor]], int],
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        if isinstance(data_source, IterableDataset):
            raise TypeError("Currently does not support IterableDataset!")

        self.data_source = data_source
        self.seq_len_fn = seq_len_fn
        self.bucket_boundaries = bucket_boundaries + [float("inf")]
        self.batch_size = batch_size
        if shuffle:
            self.sampler = RandomSampler(data_source)
        else:
            self.sampler = SequentialSampler(data_source)

        self.buckets = []
        for _ in range(len(bucket_boundaries) + 1):
            self.buckets.append([])

    def __iter__(self):
        for idx in self.sampler:
            row = self.data_source[idx]
            for bidx, boundary in enumerate(self.bucket_boundaries):
                if self.seq_len_fn(row) <= boundary:
                    self.buckets[bidx].append(idx)
                    break
            # Flush the buckets
            for bidx, bucket in enumerate(self.buckets):
                if len(bucket) == self.batch_size:
                    yield sorted(
                        bucket, key=lambda ix: -self.seq_len_fn(self.data_source[ix])
                    )
                    self.buckets[bidx] = []
        # Flush leftovers
        for bidx, bucket in enumerate(self.buckets):
            if len(bucket) > 0:
                yield sorted(
                    bucket, key=lambda ix: -self.seq_len_fn(self.data_source[ix])
                )

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size


class BatchSampler(Sampler):
    """Defines a batch sampler that batches examples of similar lengths together and
    minimizes amount of padding needed. This BatchSampler works by initially taking a large
    steps (multiplied by 100) and then sort the data according to `seq_len_fn`.
    Arguments:
        data_source: data source to sample from.
        seq_len_fn: function to return the current length of the sequence.
        batch_size: size of mini-batch.
            Default: 32
        shuffle: data_source will be wrapped with RandomSampler if set to ``True``,
            otherwise, SequentialSampler. Default: True
    Example:
        >>> dummy = [
            torch.tensor(range(1, torch.randint(2, 11, (1,))[0])) for num in range(10)
        ]
        >>> def tensor_seq_len_fn(row):
        ...     return row.size(0)
        >>> list(BatchSampler(dummy, tensor_seq_len_fn, batch_size=5, shuffle=False))
        [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9]]
        >>> list(BatchSampler(dummy, tensor_seq_len_fn, batch_size=5))
        [[9, 2, 4, 3, 1], [8, 7, 5, 6], [0]]
    """

    def __init__(
        self,
        data_source: Dataset,
        seq_len_fn: Callable[[Union[List[Any], torch.Tensor]], int],
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        if isinstance(data_source, IterableDataset):
            raise TypeError("Currently does not support IterableDataset!")

        self.data_source = data_source
        self.seq_len_fn = seq_len_fn
        self.batch_size = batch_size
        if shuffle:
            self.sampler = RandomSampler(data_source)
        else:
            self.sampler = SequentialSampler(data_source)

    def __iter__(self):
        sample_count = 100
        minibatch = []
        for idx in self.sampler:
            if len(minibatch) % (self.batch_size * sample_count) == 0:
                for batch in self._batch(minibatch):
                    yield batch
                minibatch = []
            heappush(minibatch, (self.seq_len_fn(self.data_source[idx]), idx))

        # Finish up leftovers
        if minibatch:
            for batch in self._batch(minibatch):
                yield batch

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def _batch(self, minibatch: List[Tuple[torch.Tensor, int]]):
        total_iter = (len(minibatch) + self.batch_size - 1) // self.batch_size
        for _ in range(total_iter):
            max_steps = min(self.batch_size, len(minibatch))
            # Return ordered data
            batch_iter = [heappop(minibatch) for _ in range(max_steps)]
            yield list(map(lambda x: x[1], batch_iter))
