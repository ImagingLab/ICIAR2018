import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .patch_extractor import PatchExtractor

LABELS = ['NORMAL', 'BENIGN', 'INSITU', 'INVASIVE']
IMAGE_SIZE = (2048, 1536)
PATCH_SIZE = 512


class BachDataset(Dataset):
    def __init__(self, path, stride, augment=False):
        super(Dataset, self).__init__()

        wp = int((IMAGE_SIZE[0] - PATCH_SIZE) / stride + 1)
        hp = int((IMAGE_SIZE[1] - PATCH_SIZE) / stride + 1)
        names = glob.glob(path + '/*.tif')

        self.path = path
        self.stride = stride
        self.names = names
        self.shape = (len(names), wp, hp, (4 if augment else 1), (2 if augment else 1))  # (files, x_patches, y_patches, rotations, flip)

    def __getitem__(self, index):
        img, xpatch, ypatch, rotation, flip = np.unravel_index(index, self.shape)
        ex = PatchExtractor(path=self.names[img], patch_size=PATCH_SIZE, stride=self.stride)
        patch = ex.extract_patch((xpatch, ypatch))

        if rotation != 0:
            patch = patch.rotate(rotation * 3)

        if flip != 0:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)

        return patch

    def __len__(self):
        return np.prod(self.shape)
