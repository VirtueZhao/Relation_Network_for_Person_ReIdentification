import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import Sampler

class TripletBatchSampler(Sampler):
    def __init__(self, data_source, batch_image=4):
        super(TripletBatchSampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.batch_image = batch_image

        print("Test Triplet Batch Sampler")

        # Sort by PID
        indices = np.argsort(np.asarray(data_source)[:, 1])
        self.index_map = dict(zip(np.arange(self.num_samples), indices))

        # Get the Range of Indices for Each PID
        self.index_range = defaultdict(lambda: [self.num_samples, -1])
        for i, j in enumerate(indices):
            _, pid, _ = data_source[j]
            self.index_range[pid][0] = min(self.index_range[pid][0], i)
            self.index_range[pid][1] = max(self.index_range[pid][1], i)

        print("--Test--")
        # for k in self.index_map:
        #     print("Key: {}. Value: {}".format(k, self.index_map[k]))
        indices = np.random.permutation(self.num_samples)
        for i in indices:
            anchor_index = self.index_map[i]
            print("Indices i: {}".format(i))
            print("Anchor Index: {}".format(anchor_index))
            _, pid, _ = self.data_source[anchor_index]
            print("PID: {}".format(pid))
            start, end = self.index_range[pid]
            print("Start: {}".format(start))
            print("End: {}".format(end))
            pos_indices = _choose_from(start, end, excluded_range=(i, i), size=self.batch_image)

            exit()


    def __iter__(self):
        indices = np.random.permutation(self.num_samples)

    def __len__(self):
        return self.num_samples * self.batch_image
