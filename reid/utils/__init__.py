import torch


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot Convert {} to Torch Tensor.".format(type(ndarray)))

    return ndarray
