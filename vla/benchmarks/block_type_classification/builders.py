import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

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
    def __init__(self, directory: str, dataset_size: int = 200, train_split: float = 0.25, random_seed=None, full_folder=False):
        self.directory = directory
        self.dataset_size = dataset_size
        self.train_split = train_split
        self.random_seed = random_seed
        self.full_folder = full_folder
        if self.random_seed is not None:
            random.seed(self.random_seed)

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
    def build(self, json_name: str) -> Tuple[str, str, list, list]:
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

    def build(self, json_name: str) -> Tuple[str, str, list, list]:
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

        if self.full_folder:
            gen_set = []
            for los, path in samples:
                gen_set.append((los["distance"], path))
            return "", "", gen_set, []
        
        train_sz = int(self.train_split * usable_per_bin)
        test_sz = usable_per_bin

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
            test_set.extend(bucket[train_sz:test_sz])

        if self.random_seed is not None:
            return "", "", train_set, test_set

        train_file = f"train_{json_name}.json"
        test_file = f"test_{json_name}.json"

        json.dump(train_set, open(train_file, "w"))
        json.dump(test_set, open(test_file, "w"))

        return train_file, test_file, [], []


# Type Dataset Builder


class TypeDatasetBuilder(DatasetBuilder):

    def build(self, json_name: str) -> Tuple[str, str, list, list]:
        samples = self.extract_samples()
        if not samples:
            raise Exception("No valid samples for type dataset.")

        type_map: Dict[str, List[str]] = {}

        for los, path in samples:
            label = los["type"]
            type_map.setdefault(label, []).append(path)

        if self.full_folder:
            train_sz = None
            per_class = 0
        else:
            per_class = self.dataset_size
            train_sz = int(self.train_split * per_class)

        train_out, test_out = {}, {}

        for label, paths in type_map.items():
            if len(paths) < per_class:
                continue
            random.shuffle(paths)
            train_out[label] = paths[:train_sz]
            test_out[label] = paths[train_sz:per_class]

        if self.random_seed is not None:
            return "", "", train_out, test_out

        train_file = f"train_{json_name}.json"
        test_file = f"test_{json_name}.json"

        json.dump(train_out, open(train_file, "w"))
        json.dump(test_out, open(test_file, "w"))

        return train_file, test_file, [], []
