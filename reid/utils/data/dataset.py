import numpy as np
import os.path as osp
from tabulate import tabulate
from ..serialization import read_json


def _pluck(identities, indices, relabel=False):
    ret = []
    query = {}

    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        if relabel:
            if index not in query.keys():
                query[index] = []
        else:
            if pid not in query.keys():
                query[pid] = []
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                # name = osp.splitext(fname)[0]
                # x, y, _ = map(int, name.split('_'))
                if relabel:
                    ret.append((fname, index, camid))
                    query[index].append(fname)
                else:
                    ret.append((fname, pid, camid))
                    query[pid].append(fname)

    return ret, query


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
        self.val, self.val_query = _pluck(identities, val_pids, relabel=True)
        self.trainval, self.trainval_query = _pluck(identities, trainval_pids, relabel=True)

        self.test_list = read_json(osp.join(self.dataset_path, "test.json"))[0]
        for i in range(len(self.test_list['query'])):
            self.test_list['query'][i] = tuple(self.test_list['query'][i])
        for i in range(len(self.test_list['gallery'])):
            self.test_list['gallery'][i] = tuple(self.test_list['gallery'][i])
        self.query = self.test_list['query']
        self.gallery = self.test_list['query']
        self.num_train_ids = len(train_pids)
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = len(trainval_pids)

        if verbose:
            print("Dataset {} Loaded".format(self.__class__.__name__))

            dataset_table = [
                ["Subset", "# ID", "# Images"],
                ["Train", self.num_train_ids, len(self.train)],
                ["Val", self.num_val_ids, len(self.val)],
                ["Trainval", self.num_trainval_ids, len(self.trainval)],
                ["Query", len(self.split['query']), len(self.query)],
                ["Gallery", len(self.split['gallery']), len(self.gallery)]
            ]
            
            print(tabulate(dataset_table))


            # print("  query    | {:5d} | {:8d}"
            #       .format(len(self.split['query']), len(self.query)))
            # print("  gallery  | {:5d} | {:8d}"
            #       .format(len(self.split['gallery']), len(self.gallery)))





    def _check_integrity(self):
        return osp.isdir(osp.join(self.dataset_path, "images")) and \
            osp.isfile(osp.join(self.dataset_path, "meta.json")) and \
            osp.isfile(osp.join(self.dataset_path, "splits.json")) and \
            osp.isdir(osp.join(self.dataset_path, "poses"))
