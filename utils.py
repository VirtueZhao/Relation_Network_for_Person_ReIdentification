import os.path as osp
from reid import datasets
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import TripletBatchSampler

from torch.utils.data import DataLoader


def build_data_loader(dataset_name, split_id, data_path, img_height, img_width, batch_size, num_workers,
                      combine_trainval, np_ratio):

    print("Build Data Loader")
    data_path = osp.join(data_path, dataset_name)
    dataset = datasets.create(dataset_name, data_path, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train

    train_transformer = T.Compose([
        T.RectScale(img_height, img_width),
        T.RandomSizedEarser(),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer
    ])

    test_transformer = T.Compose([
        T.RectScale(img_height, img_width),
        T.ToTensor(),
        normalizer
    ])

    train_loader = DataLoader(
        Preprocessor(train_set,
                     dataset_path=dataset.images_dir, transform=train_transformer),
        sampler=TripletBatchSampler(train_set),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        Preprocessor(dataset.val,
                     dataset_path=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False
    )

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     dataset_path=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False
    )

    # t = TripletBatchSampler(train_set)

    return dataset, train_loader, val_loader, test_loader
