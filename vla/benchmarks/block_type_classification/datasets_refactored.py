import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io


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
    """
    Abstract class for dataset creators.
    """

    def __init__(self, directory: str, dataset_size: int = 200, train_split: float = 0.25):
        self.directory = directory
        self.dataset_size = dataset_size
        self.train_split = train_split

    # Common Logic

    def extract_samples(self) -> List[Tuple[dict, str]]:
        """Find all images with valid JSON metadata."""
        samples = []
        basepath = Path(self.directory)

        for jpg_path in basepath.rglob("*.jpg"):
            path_stem = str(jpg_path.with_suffix(""))
            json_path = path_stem + ".json"

            meta = safe_load_json(json_path)
            if meta is None:
                continue

            los = meta.get("lineOfSight", {})
            if los.get("hitType") == "MISS":
                continue

            samples.append((los, path_stem))

        return samples

    # Abstract: each builder must implement

    @abstractmethod
    def build(self, json_name: str) -> None:
        ...


# Distance Dataset Builder

class DistanceDatasetBuilder(DatasetBuilder):
    def __init__(self, directory: str, num_bins: int = 50, **kwargs):
        super().__init__(directory, **kwargs)
        self.num_bins = num_bins

    def build(self, json_name: str) -> None:
        samples = self.extract_samples()
        if not samples:
            raise Exception("No valid samples for distance dataset.")

        distances = [s[0]["distance"] for s in samples]

        # Histogram
        hist_counts, bin_edges = np.histogram(distances, bins=self.num_bins)
        mid = self.num_bins // 2

        bin_edges = bin_edges[:mid + 1]
        usable_count = hist_counts[mid]
        usable = min(self.dataset_size, usable_count)

        train_sz = int(self.train_split * usable)
        test_sz = usable - train_sz

        # Bucket by bins
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
            if len(bucket) < usable:
                continue
            random.shuffle(bucket)
            train_set.extend(bucket[:train_sz])
            test_set.extend(bucket[train_sz: train_sz + test_sz])

        json.dump(train_set, open(f"train_{json_name}.json", "w"))
        json.dump(test_set, open(f"test_{json_name}.json", "w"))


# Type Dataset Builder

class TypeDatasetBuilder(DatasetBuilder):
    def build(self, json_name: str) -> None:
        samples = self.extract_samples()
        if not samples:
            raise Exception("No valid samples for type dataset.")

        type_map: Dict[str, List[str]] = {}

        for los, path in samples:
            t = los["type"]
            type_map.setdefault(t, []).append(path)

        train_set, test_set = {}, {}

        per_class = self.dataset_size
        train_sz = int(self.train_split * per_class)

        for label, paths in type_map.items():
            if len(paths) < per_class:
                continue
            random.shuffle(paths)
            train_set[label] = paths[:train_sz]
            test_set[label] = paths[train_sz:per_class]

        json.dump(train_set, open(f"train_{json_name}.json", "w"))
        json.dump(test_set, open(f"test_{json_name}.json", "w"))


# PyTorch Dataset

class LOSDataset(Dataset):
    def __init__(self, items, num_classes=None, transform=None):
        """
        items:
            classification -> list of (class_idx, path_stem)
            regression     -> list of (distance, path_stem)
        """
        self.items = items
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def label_tensor(self, value):
        if self.num_classes is None:  # regression
            return torch.tensor([float(value)], dtype=torch.float32)

        y = torch.zeros(self.num_classes)
        y[value] = 1.0
        return y

    def __getitem__(self, idx):
        label, path_stem = self.items[idx]

        image = io.imread(path_stem + ".jpg")
        if self.transform:
            image = self.transform(image)

        return self.label_tensor(label), image


# Dataset Manager (Main Facade)


class DatasetManager:
    """
    High-level interface:
        - create dataset JSONs
        - load them into PyTorch DataLoaders
    """

    def __init__(
        self,
        directory: str,
        label_type: str,                # "distance" or "type"
        dataset_size: int = 200,
        train_split: float = 0.25,
        num_bins: int = 50,
    ):
        self.directory = directory
        self.label_type = label_type
        self.dataset_size = dataset_size
        self.train_split = train_split
        self.num_bins = num_bins

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
            raise ValueError(f"Unknown label_type '{label_type}'")

    # Dataset creation

    def create(self, json_name: str):
        self.builder.build(json_name)

    # Loader utilities

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

    # DataLoader creation

    def make_dataloaders(self, train_json: str, test_json: str, transform=None,
                         batch_size=32, num_workers=2):

        train_raw = self.read_json(train_json)
        test_raw = self.read_json(test_json)

        if self.label_type == "type":  # classification
            train_items, num_classes = self.flatten_class_map(train_raw)
            test_items, _ = self.flatten_class_map(test_raw)

        else:  # regression
            train_items = train_raw
            test_items = test_raw
            num_classes = None

        train_ds = LOSDataset(train_items, num_classes, transform)
        test_ds = LOSDataset(test_items, num_classes, transform)

        train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers)
        score_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        # num_classes defaults to 1 for regression
        return train_loader, test_loader, score_loader, (num_classes or 1)
