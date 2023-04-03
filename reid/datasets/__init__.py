from .market1501 import Market1501


AVAILABLE_DATASETS = {
    "market1501": Market1501
}


def create(dataset_name, dataset_path, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    dataset_name : str
        The dataset name. Can be one of 'cuhk03',
        'market1501', and 'dukemtmc'.
    dataset_path : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """

    if dataset_name not in AVAILABLE_DATASETS:
        raise KeyError("Unknown Dataset: ", dataset_name)
    return AVAILABLE_DATASETS[dataset_name](dataset_path, *args, **kwargs)
