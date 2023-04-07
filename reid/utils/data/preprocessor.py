import os.path as osp
from PIL import Image
from reid.utils.data import transforms


class Preprocessor(object):
    def __init__(self, dataset, dataset_path=None, with_pose=False, pose_root=None, pid_imgs=None, height=256, width=128, pose_aug='no', transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.with_pose = with_pose
        self.pose_root = pose_root
        self.pid_imgs = pid_imgs
        self.height = height
        self.width = width
        self.pose_aug = pose_aug

        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if transform == None:
            self.transform = transforms.Compose([
                transforms.RectScale(height, width),
                transforms.RandomSizedEarser(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalizer
            ])
        else:
            self.transform = transform
        self.transform_p = transforms.Compose([
            transforms.RectScale(height, width),
            transforms.ToTensor(),
            normalizer
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]

        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.dataset_path is not None:
            fpath = osp.join(self.dataset_path, fname)
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)
        return img, fname, pid, camid
