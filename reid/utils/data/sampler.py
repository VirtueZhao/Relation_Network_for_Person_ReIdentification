import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import Sampler


def _choose_from(start, end, excluded_range=None, size=1, replace=False):
    num = end - start + 1
    if num <= size:
        replace = True
    if excluded_range is None:
        return np.random.choice(num, size=size, replace=replace) + start
    ex_start, ex_end = excluded_range
    num_ex = ex_end - ex_start + 1
    num -= num_ex
    inds = np.random.choice(num, size, replace=replace) + start
    inds += (inds >= ex_start) * num_ex
    return inds


class TripletBatchSampler(Sampler):
    def __init__(self, data_source, batch_image=4):
        super(TripletBatchSampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.batch_image = batch_image

        # Sort by PID
        indices = np.argsort(np.asarray(data_source)[:, 1])
        self.index_map = dict(zip(np.arange(self.num_samples), indices))

        # Get the Range of Indices for Each PID
        self.index_range = defaultdict(lambda: [self.num_samples, -1])
        for i, j in enumerate(indices):
            _, pid, _ = data_source[j]
            self.index_range[pid][0] = min(self.index_range[pid][0], i)
            self.index_range[pid][1] = max(self.index_range[pid][1], i)

    def __iter__(self):
        indices = np.random.permutation(self.num_samples)
        for i in indices:
            # Anchor Sample
            anchor_index = self.index_map[i]
            _, pid, _ = self.data_source[anchor_index]
            # Positive Sample
            start, end = self.index_range[pid]
            pos_indices = _choose_from(start, end, excluded_range=(i, i), size=self.batch_image)
            for pos_index in pos_indices:
                yield self.index_map[pos_index]

    def __len__(self):
        return self.num_samples * self.batch_image
