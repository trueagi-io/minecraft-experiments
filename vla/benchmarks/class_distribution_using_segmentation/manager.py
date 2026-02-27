import json
from torch.utils.data import DataLoader
from builders import ClassDistributionDatasetBuilder
from dataset import LOSSegmDataset
import numpy as np


class DatasetManager:
    def __init__(self, directory, dataset_size=200, train_split=0.25,
                 random_seed=None, full_folder=False):
        self.random_seed = random_seed

        self.builder = ClassDistributionDatasetBuilder(directory, dataset_size=dataset_size,
                                          train_split=train_split, random_seed=self.random_seed,
                                          full_folder=full_folder)

    def create(self, name):
        if self.random_seed is not None:
            _, _, train_set, test_set = self.builder.build(name)
            return train_set, test_set
        train_file, test_file, [], [] = self.builder.build(name)
        print(f"Train set: {train_file}")
        print(f"Test  set: {test_file}")
        return train_file, test_file


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_class_map(mapping, pre_classes=None):
    classes = list(mapping.keys()) if pre_classes is None else pre_classes
    flat = []
    for idx, c in enumerate(classes):
        if c not in mapping:
            continue
        for p in mapping[c]:
            flat.append((idx, p))
    return flat, len(classes)


def make_dataloaders(train_json, test_json, transform=None, feature_store=None, batch=32, workers=2, dataset_manager=None,
                     generalization_dataset_manager=None):

    if dataset_manager is not None:
        train_items, test_items = dataset_manager.create("")
    else:
        train_items = load_json(train_json)
        test_items = load_json(test_json)

    generalization_items = []
    if generalization_dataset_manager is not None:
        _, generalization_items = generalization_dataset_manager.create("")

    nc = len(np.load(train_items[0][1]))

    train_ds = LOSSegmDataset(train_items, transform, feature_store)
    test_ds = LOSSegmDataset(test_items, transform, feature_store)
    generalization_ds = LOSSegmDataset(generalization_items, transform, feature_store)

    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers),
        DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=workers),
        DataLoader(test_ds, batch_size=1, shuffle=False),
        DataLoader(generalization_ds, batch_size=1, shuffle=False),
        nc
    )
