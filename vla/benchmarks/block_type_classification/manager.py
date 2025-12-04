import json
from torch.utils.data import DataLoader
from builders import DistanceDatasetBuilder, TypeDatasetBuilder
from dataset import LOSDataset
from config import LabelType


class DatasetManager:
    def __init__(self, directory, label_type: LabelType, dataset_size=200, train_split=0.25, num_bins=50, random_seed=None):
        self.random_seed = random_seed
        self.label_type = label_type

        if label_type == LabelType.DISTANCE:
            self.builder = DistanceDatasetBuilder(directory, dataset_size=dataset_size,
                                                  train_split=train_split, num_bins=num_bins, random_seed=self.random_seed)
        elif label_type == LabelType.TYPE_CLASSIFICATION:
            self.builder = TypeDatasetBuilder(directory, dataset_size=dataset_size,
                                              train_split=train_split, random_seed=self.random_seed)
        else:
            raise ValueError(f"Unsupported label type: {label_type}")

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


def flatten_class_map(mapping):
    classes = list(mapping.keys())
    flat = []
    for idx, c in enumerate(classes):
        for p in mapping[c]:
            flat.append((idx, p))
    return flat, len(classes)


def make_dataloaders(train_json, test_json, label_type: LabelType, transform=None, feature_store=None, batch=32, workers=2, dataset_manager=None):

    if dataset_manager is not None:
        train_raw, test_raw = dataset_manager.create("")
    else:
        train_raw = load_json(train_json)
        test_raw = load_json(test_json)

    if label_type == LabelType.TYPE_CLASSIFICATION:
        train_items, nc = flatten_class_map(train_raw)
        test_items, _ = flatten_class_map(test_raw)
    elif label_type == LabelType.DISTANCE:
        train_items = train_raw
        test_items = test_raw
        nc = None
    else:
        raise ValueError(f"Unsupported label type: {label_type}")

    train_ds = LOSDataset(train_items, nc, transform, feature_store)
    test_ds = LOSDataset(test_items, nc, transform, feature_store)

    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers),
        DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=workers),
        DataLoader(test_ds, batch_size=1, shuffle=False),
        nc if nc else 1
    )
