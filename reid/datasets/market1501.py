from ..utils.data import Dataset


class Market1501(Dataset):

    def __init__(self, root, split_id=0, num_val=100):
        super(Market1501, self).__init__(root, split_id=split_id)

        if not self._check_integrity():
            raise RuntimeError("Dataset Not Found or Corrupted. Please Follow README.md to Prepare Market1501 Dataset")

        self.load(num_val)
