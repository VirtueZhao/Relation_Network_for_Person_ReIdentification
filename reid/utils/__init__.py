import torch


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numoy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot Convert {} to Numpy Array".format(type(tensor)))

    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot Convert {} to Torch Tensor.".format(type(ndarray)))

    return ndarray
