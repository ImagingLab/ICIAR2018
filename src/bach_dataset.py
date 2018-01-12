import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from .patch_extractor import PatchExtractor

LABELS = ['Normal', 'Benign', 'InSitu', 'Invasive']
IMAGE_SIZE = (2048, 1536)
PATCH_SIZE = 512


class BachDataset(Dataset):
    def __init__(self, path, stride=PATCH_SIZE, augment=False):
        super(Dataset, self).__init__()

        wp = int((IMAGE_SIZE[0] - PATCH_SIZE) / stride + 1)
        hp = int((IMAGE_SIZE[1] - PATCH_SIZE) / stride + 1)
        labels = {name: index for index in range(len(LABELS)) for name in glob.glob(path + '/' + LABELS[index] + '/*.tif')}

        self.path = path
        self.stride = stride
        self.labels = labels
        self.names = list(sorted(labels.keys()))
        self.shape = (len(labels), wp, hp, (4 if augment else 1), (2 if augment else 1))  # (files, x_patches, y_patches, rotations, flip)

    def __getitem__(self, index):
        img, xpatch, ypatch, rotation, flip = np.unravel_index(index, self.shape)
        extractor = PatchExtractor(path=self.names[img], patch_size=PATCH_SIZE, stride=self.stride)
        patch = extractor.extract_patch((xpatch, ypatch))

        if rotation != 0:
            patch = patch.rotate(rotation * 90)

        if flip != 0:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)

        label = self.labels[self.names[img]]

        return transforms.ToTensor()(patch), label

    def __len__(self):
        return np.prod(self.shape)
