import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from PIL import Image

# PyTorch Dataset


class LOSSegmDataset(Dataset):
    def __init__(self, items, transform=None, feature_store=None):
        self.items = items
        self.transform = transform
        self.feature_store = feature_store

    def __len__(self):
        return len(self.items)

    def load_image(self, img_path):
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        img_path, distr_path = self.items[idx]
        distribution = torch.from_numpy(np.load(distr_path))
        feature_path = str(img_path).strip(".png").replace("/", "_")
        if self.feature_store and self.feature_store.exists(feature_path):
            features = self.feature_store.load(feature_path)
            return distribution, features

        return distribution, self.load_image(img_path)
