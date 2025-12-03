import json
import os
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io

from PIL import Image


# Helpers


def safe_load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


# Base Builder (Abstract)


class DatasetBuilder(ABC):
    def __init__(self, directory: str, dataset_size: int = 200, train_split: float = 0.25):
        self.directory = directory
        self.dataset_size = dataset_size
        self.train_split = train_split

    def extract_samples(self) -> List[Tuple[dict, str]]:
        """Gather (lineOfSight_metadata, image_stem_path)."""
        samples = []
        basepath = Path(self.directory)

        for jpg_path in basepath.rglob("*.jpg"):
            stem = str(jpg_path.with_suffix(""))
            json_path = stem + ".json"

            meta = safe_load_json(json_path)
            if meta is None:
                continue

            los = meta.get("lineOfSight", {})
            if los.get("hitType") == "MISS":
                continue

            samples.append((los, stem))

        return samples

    @abstractmethod
    def build(self, json_name: str) -> Tuple[str, str]:
        """
        Must return:
            (train_json_path, test_json_path)
        """
        ...


# Distance Dataset Builder


class DistanceDatasetBuilder(DatasetBuilder):
    def __init__(self, directory: str, num_bins: int = 50, **kwargs):
        super().__init__(directory, **kwargs)
        self.num_bins = num_bins

    def build(self, json_name: str) -> Tuple[str, str]:
        samples = self.extract_samples()
        if not samples:
            raise Exception("No valid samples for distance dataset.")

        distances = [s[0]["distance"] for s in samples]
        hist_counts, bin_edges = np.histogram(distances, bins=self.num_bins)
        mid = self.num_bins // 2

        # Take only the first half of bins
        bin_edges = bin_edges[:mid + 1]
        usable_count = hist_counts[mid]
        usable_per_bin = min(self.dataset_size, usable_count)

        train_sz = int(self.train_split * usable_per_bin)
        test_sz = usable_per_bin - train_sz

        # Bucket samples
        buckets = {i: [] for i in range(mid)}
        for los, path in samples:
            dist = los["distance"]
            if dist > bin_edges[-1]:
                continue
            for i in range(mid):
                if dist < bin_edges[i + 1]:
                    buckets[i].append((dist, path))
                    break

        train_set, test_set = [], []

        for bucket in buckets.values():
            if len(bucket) < usable_per_bin:
                continue
            random.shuffle(bucket)
            train_set.extend(bucket[:train_sz])
            test_set.extend(bucket[train_sz: train_sz + test_sz])

        train_file = f"train_{json_name}.json"
        test_file = f"test_{json_name}.json"

        json.dump(train_set, open(train_file, "w"))
        json.dump(test_set, open(test_file, "w"))

        return train_file, test_file


# Type Dataset Builder


class TypeDatasetBuilder(DatasetBuilder):
    def build(self, json_name: str) -> Tuple[str, str]:
        samples = self.extract_samples()
        if not samples:
            raise Exception("No valid samples for type dataset.")

        type_map: Dict[str, List[str]] = {}

        for los, path in samples:
            label = los["type"]
            type_map.setdefault(label, []).append(path)

        per_class = self.dataset_size
        train_sz = int(self.train_split * per_class)

        train_out, test_out = {}, {}

        for label, paths in type_map.items():
            if len(paths) < per_class:
                continue
            random.shuffle(paths)
            train_out[label] = paths[:train_sz]
            test_out[label] = paths[train_sz:per_class]

        train_file = f"train_{json_name}.json"
        test_file = f"test_{json_name}.json"

        json.dump(train_out, open(train_file, "w"))
        json.dump(test_out, open(test_file, "w"))

        return train_file, test_file


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


# DatasetManager


class DatasetManager:
    def __init__(
        self,
        directory: str,
        label_type: str,     # "distance" or "type"
        dataset_size: int = 200,
        train_split: float = 0.25,
        num_bins: int = 50
    ):
        self.label_type = label_type

        if label_type == "distance":
            self.builder = DistanceDatasetBuilder(
                directory,
                dataset_size=dataset_size,
                train_split=train_split,
                num_bins=num_bins
            )
        elif label_type == "type":
            self.builder = TypeDatasetBuilder(
                directory,
                dataset_size=dataset_size,
                train_split=train_split
            )
        else:
            raise ValueError(f"Unknown label_type: {label_type}")

    def create(self, json_name: str, verbose=True):
        start = time.time()

        train_file, test_file = self.builder.build(json_name)

        elapsed = time.time() - start

        if verbose:
            print(f"Dataset generation completed in {elapsed:.3f} seconds")
            print(f"    Train: {train_file}")
            print(f"    Test : {test_file}")

        return train_file, test_file, elapsed

    @staticmethod
    def read_json(path: str):
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def flatten_class_map(mapping: Dict[str, list]):
        classes = list(mapping.keys())
        flat = []
        for label in classes:
            idx = classes.index(label)
            for p in mapping[label]:
                flat.append((idx, p))
        return flat, len(classes)

    def make_dataloaders(self, train_json, test_json, transform=None,
                         batch_size=32, num_workers=0):

        train_raw = self.read_json(train_json)
        test_raw = self.read_json(test_json)

        if self.label_type == "type":
            train_items, num_classes = self.flatten_class_map(train_raw)
            test_items, _ = self.flatten_class_map(test_raw)
        else:
            train_items = train_raw
            test_items = test_raw
            num_classes = None

        train_ds = LOSDataset(train_items, num_classes, transform)
        test_ds = LOSDataset(test_items, num_classes, transform)

        train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers)
        score_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        return train_loader, test_loader, score_loader, (num_classes or 1)

def CreateDataloader(train_json_file, test_json_file, preprocessor,
                     directory=None, label_type=None, dataset_size=200,
                     train_split=0.25, num_bins=50):

    """
    This function provides backwards compatibility with the old API.
    `directory` and `label_type` must be provided if classification/regression
    must be determined dynamically. If not, you can create a DatasetManager
    externally and call make_dataloaders directly.
    """

    if label_type is None:
        raise ValueError("label_type must be specified for CreateDataloader()")

    manager = DatasetManager(
        directory=directory,
        label_type=label_type,
        dataset_size=dataset_size,
        train_split=train_split,
        num_bins=num_bins
    )

    train_loader, test_loader, score_loader, num_classes = manager.make_dataloaders(
        train_json_file,
        test_json_file,
        transform=preprocessor
    )

    return train_loader, test_loader, score_loader, num_classes, label_type

manager = DatasetManager(
    directory="../../../Mountain_Range",
    label_type="distance",
    dataset_size=200,
    train_split=0.25,
    num_bins=50
)

train_file, test_file, t = manager.create("los_dataset")

print("Generation time:", t)
