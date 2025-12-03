import os
import torch
from pathlib import Path
from tqdm import tqdm

class FeatureStore:
    """
    Saves and loads precomputed features to avoid running DINO inside dataloaders.
    """

    def __init__(self, root="./precomputed_features"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def feature_path(self, img_stem: str) -> Path:
        return self.root / f"{Path(img_stem).name}.pt"

    def exists(self, img_stem: str) -> bool:
        return self.feature_path(img_stem).exists()

    def save(self, img_stem: str, tensor: torch.Tensor):
        torch.save(tensor.cpu(), self.feature_path(img_stem))

    def load(self, img_stem: str) -> torch.Tensor:
        return torch.load(self.feature_path(img_stem))