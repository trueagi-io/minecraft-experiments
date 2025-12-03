import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image

# PyTorch Dataset


class LOSDataset(Dataset):
    def __init__(self, items, num_classes=None, transform=None):
        self.items = items
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def label_tensor(self, val):
        if self.num_classes is None:
            return torch.tensor([float(val)], dtype=torch.float32)
        y = torch.zeros(self.num_classes)
        y[val] = 1.0
        return y

    def __getitem__(self, idx):
        label, path_stem = self.items[idx]
        img_path = path_stem + ".jpg"

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return self.label_tensor(label), image
