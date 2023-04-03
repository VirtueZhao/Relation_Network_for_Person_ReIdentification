import os.path as osp
from reid import datasets

def build_data_loader(dataset_name, split_id, data_path, img_height, img_width, batch_size, num_workers,
                      combine_trainval, np_ratio):

    print("Build Data Loader")
    data_path = osp.join(data_path, dataset_name)
    dataset = datasets.create(dataset_name, data_path, split_id=split_id)

    return 1, 2, 3, 4
