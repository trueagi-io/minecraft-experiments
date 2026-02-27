import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np
from tqdm import tqdm

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

    @abstractmethod
    def extract_samples(self) -> List[Tuple[List, str]]:
        """
        Must return:
            [([class_distribution_probs], image_path)]
        """
        ...

    @abstractmethod
    def build(self, json_name: str) -> Tuple[str, str, list, list]:
        """
        Must return:
            (train_json_path, test_json_path)
        """
        ...

# Class Distribution Builder

class ClassDistributionDatasetBuilder(DatasetBuilder):

    def extract_samples(self) -> List[Tuple[List, str]]:
        """Gather (class_probs, image_stem_path)."""
        paths = []
        basepath = Path(self.directory)
        images_subfolders = list(basepath.rglob("images"))

        for img_folder_path in images_subfolders:
            distrs_path = img_folder_path.parent / "distributions"
            images_folder = list(img_folder_path.rglob("*.png"))
            for img_path in images_folder:
                distr_path = (distrs_path / img_path.stem).with_suffix(".npy")
                if not distr_path.exists():
                    continue
                paths.append((img_path, distr_path))
        return paths

    def build(self, json_name: str) -> Tuple[str, str, list, list]:
        samples = self.extract_samples()
        if not samples:
            raise Exception("No valid samples for type dataset.")

        if self.full_folder:
            train_split = 0
        else:
            train_split = int(self.train_split * len(samples))

        random.shuffle(samples)

        train_out = samples[:train_split]
        test_out = samples[train_split:]

        if self.random_seed is not None:
            return "", "", train_out, test_out

        train_file = f"train_{json_name}.json"
        test_file = f"test_{json_name}.json"

        json.dump(train_out, open(train_file, "w"))
        json.dump(test_out, open(test_file, "w"))

        return train_file, test_file, [], []
