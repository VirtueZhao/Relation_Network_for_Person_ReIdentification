import os.path as osp

import numpy as np

from ..serialization import read_json


class Dataset(object):
    def __init__(self, dataset_path, split_id=0):
        self.dataset_path = dataset_path
        self.split_id = split_id
        self.meta = None
        self.split = None
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    def load(self, num_val=0.3, verbose=True):
        splits = read_json(osp.join(self.dataset_path, "splits.json"))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}".format(len(splits)))

        self.split = splits[self.split_id]

        trainval_pids = sorted(np.asarray(self.split['trainval']))
        num = len(trainval_pids)
        if isinstance(num_val, float):
            num_val = int(round(num * num_val))
        if num_val >= num or num_val < 0:
            raise ValueError("num_val exceeds total identities {}".format(num))
        train_pids = sorted(trainval_pids[:-num_val])
        val_pids = sorted(trainval_pids[-num_val:])

        self.meta = read_json(osp.join(self.dataset_path, "meta.json"))
        identities = self.meta["identities"]
        self.train, self.train_query = _pluck(identities, train_pids, relabel=True)




    def _check_integrity(self):
        return osp.isdir(osp.join(self.dataset_path, "images")) and \
            osp.isfile(osp.join(self.dataset_path, "meta.json")) and \
            osp.isfile(osp.join(self.dataset_path, "splits.json")) and \
            osp.isdir(osp.join(self.dataset_path, "poses"))
