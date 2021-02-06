from torch.utils.data import Dataset
import os
import json
import imutils
from imutils import paths
import numpy as np
from PIL import Image
import cv2


class CustomDataset(Dataset):
    def __init__(self, image_path, transform=None):
        # self.root_dir = root_dir
        self.transform = transform
        self.data_path = [image_path]

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        path = self.data_path[index]

        img = cv2.imread(path)
        img = np.array(img)

        if self.transform:
            img = self.transform(img)
        return img
