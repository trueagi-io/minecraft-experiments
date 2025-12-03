import json
from torch.utils.data import DataLoader
from builders import DistanceDatasetBuilder, TypeDatasetBuilder
from dataset import LOSDataset
from config import LabelType


class DatasetManager:
    def __init__(self, directory, label_type: LabelType, dataset_size=200, train_split=0.25, num_bins=50):
        self.label_type = label_type

        if label_type == LabelType.DISTANCE:
            self.builder = DistanceDatasetBuilder(directory, dataset_size=dataset_size,
                                                  train_split=train_split, num_bins=num_bins)
        elif label_type == LabelType.TYPE_CLASSIFICATION:
            self.builder = TypeDatasetBuilder(directory, dataset_size=dataset_size,
                                              train_split=train_split)
        else:
            raise ValueError(f"Unsupported label type: {label_type}")

    def create(self, name):
        train_file, test_file = self.builder.build(name)
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


def make_dataloaders(train_json, test_json, label_type: LabelType, transform=None, batch=32, workers=2):
    train_raw = load_json(train_json)
    test_raw = load_json(test_json)

    if label_type == LabelType.TYPE_CLASSIFICATION:
        train_items, nc = flatten_class_map(train_raw)
        test_items, _ = flatten_class_map(test_raw)
    elif label_type == LabelType.TYPE_CLASSIFICATION:
        train_items = train_raw
        test_items = test_raw
        nc = None
    else:
        raise ValueError(f"Unsupported label type: {label_type}")

    train_ds = LOSDataset(train_items, nc, transform)
    test_ds = LOSDataset(test_items, nc, transform)

    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers),
        DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=workers),
        DataLoader(test_ds, batch_size=1, shuffle=False),
        nc if nc else 1
    )
