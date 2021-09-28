import os

import numpy as np
import pandas as pd

import torch

from torch.utils.data import Dataset

PROCESSED_DATA_PATH = 'processed_data/'


class CustomDataset(Dataset):
    def __init__(self, data_file, label_encoding, transform=None):
        self.data = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, data_file)).sample(frac=1).reset_index(drop=True)
        self.label_encoding = label_encoding
        self.transform = transform

    def __getitem__(self, indx):
        img_path, status = self.data.iloc[indx]

        image = np.load(os.path.join(PROCESSED_DATA_PATH, img_path))
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)

        label = self.label_encoding[status]
        label = torch.tensor(label).long()

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data)
