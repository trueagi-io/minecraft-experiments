from pathlib import Path
from skimage import io
import json
import os
import random
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

# LABEL_TYPE = "type"      # classification
LABEL_TYPE = "distance"   # regression
NUM_OF_BINS = 50 # used if label is "distance" to create distance histogram

DATASET_SIZE = 200
TRAIN_NUMBER = int(DATASET_SIZE * 0.25)
TEST_NUMBER = DATASET_SIZE - TRAIN_NUMBER
BATCH_SIZE = 32
NUM_WORKERS = 2

def make_dataset_distance(directory_path, json_name):
    base_path = Path(directory_path)
    jpeg_files = list(base_path.rglob('*.jpg'))
    distances = []
    full_dataset = []
    for jpeg_file in jpeg_files:
        file_name = f"./{str(jpeg_file.parent)}/{jpeg_file.stem}"
        json_path = f"{file_name}.json"
        if not os.path.exists(json_path):
            jpeg_file.unlink()
            continue
        with open(json_path, "r") as file:
            data = json.load(file)
            if data["lineOfSight"]['hitType'] == 'MISS':
                continue
            distances.append(data["lineOfSight"]["distance"])
            full_dataset.append((data["lineOfSight"]["distance"], file_name))
    hist_counts, bin_edges = np.histogram(distances, bins=NUM_OF_BINS)
    middle_bin = int(NUM_OF_BINS / 2)
    bin_edges = bin_edges[:middle_bin+1]
    middle_count = hist_counts[middle_bin]
    dataset_size = DATASET_SIZE if DATASET_SIZE < middle_count else middle_count
    train_size = int(dataset_size * 0.25)
    test_size = dataset_size - train_size
    dataset = dict(zip(list(range(0, middle_bin)), [[] for _ in range(middle_bin)]))
    for item in full_dataset:
        current_dist = item[0]
        if (current_dist > bin_edges[-1]):
            continue
        for bin_idx in range(middle_bin):
            if current_dist < bin_edges[bin_idx+1]:
                dataset[bin_idx].append((current_dist, item[1]))
                break
    train_dataset = []
    test_dataset = []
    for key in dataset:
        random.shuffle(dataset[key])
        train_dataset.extend(dataset[key][:train_size])
        test_dataset.extend(dataset[key][train_size:train_size+test_size])
    json.dump(train_dataset, open(f"train_{json_name}.json", 'w'))
    json.dump(test_dataset, open(f"test_{json_name}.json", 'w'))

def make_dataset_blocktype(directory_path, json_name):
    base_path = Path(directory_path)
    jpeg_files = list(base_path.rglob('*.jpg'))
    block_types = {}
    train_set = {}
    test_set = {}
    for jpeg_file in jpeg_files:
        file_name = f"./{str(jpeg_file.parent)}/{jpeg_file.stem}"
        json_path = f"{file_name}.json"
        if not os.path.exists(json_path):
            jpeg_file.unlink()
            continue
        with open(json_path, "r") as file:
            data = json.load(file)
            if data["lineOfSight"]['hitType'] == 'MISS':
                continue
            block_type_LOS = data["lineOfSight"]["type"]
            if LABEL_TYPE == "distance":
                block_type_LOS = int(block_type_LOS)
            if not block_type_LOS in block_types:
                block_types[block_type_LOS] = {}
                block_types[block_type_LOS]["count"] = 1
                block_types[block_type_LOS]["pathes"] = []
                block_types[block_type_LOS]["pathes"].append(file_name)
            else:
                block_types[block_type_LOS]["count"] += 1
                block_types[block_type_LOS]["pathes"].append(file_name)
    for key in block_types:
        if block_types[key]["count"] < DATASET_SIZE:
            continue
        random.shuffle(block_types[key]["pathes"])
        train_set[key] = block_types[key]["pathes"][:TRAIN_NUMBER]
        test_set[key] = block_types[key]["pathes"][TRAIN_NUMBER:TRAIN_NUMBER + TEST_NUMBER]

    json.dump(train_set, open(f"train_{json_name}.json", 'w'))
    json.dump(test_set, open(f"test_{json_name}.json", 'w'))

# process raw data and create train and test json files.
def make_dataset(directory_path, json_name):
    if LABEL_TYPE == "type":
        make_dataset_blocktype(directory_path, json_name)
    elif LABEL_TYPE == "distance":
        make_dataset_distance(directory_path, json_name)
    else:
        raise Exception("Unknown label for dataset construction")

# load prepared json dataset
def read_dataset(path_to_dataset):
    with open(path_to_dataset, "r") as file:
        return json.load(file)

# since json dataset for each class contains list of paths to images, I prefer to unpack those lists and create
# dataset which could be easily used to create DataLoader
def unpack_dataset(dataset, classes):
    res = []
    for key in dataset:
        class_index = classes.index(key)
        for item in dataset[key]:
            res.append((class_index, item))
    return res

# Custom Dataset class to pass to DataLoader
class BlockTypeDataset(Dataset):
    def __init__(self, unpacked_dataset, num_of_classes, root_dir = "./", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.t_set = unpacked_dataset
        self.num_of_classes = num_of_classes

    def __len__(self):
        return len(self.t_set)

    def preprocess_label(self, idx):
        if self.num_of_classes is not None:
            class_idx = self.t_set[idx][0]
            one_hot_vector = np.zeros(self.num_of_classes)
            one_hot_vector[class_idx] = 1
            return torch.tensor(one_hot_vector)
        else:
            return torch.tensor(self.t_set[idx][0]).unsqueeze(0)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, f"{self.t_set[idx][1]}.jpg")
        image = io.imread(img_name)
        if self.transform is not None:
            image = self.transform(image)
        label = self.preprocess_label(idx)
        return label, image

def get_classes(train_set):
    classes = list(train_set.keys())
    return classes

def CreateDataloader(train_json_file, test_json_file, preprocessor):
    train_set = read_dataset(train_json_file)
    test_set = read_dataset(test_json_file)
    if LABEL_TYPE == "type":
        classes = get_classes(train_set)
        train_set_unpacked = unpack_dataset(train_set, classes)
        test_set_unpacked = unpack_dataset(test_set, classes)
        train_dataset = BlockTypeDataset(train_set_unpacked, len(classes), transform=preprocessor)
        test_dataset = BlockTypeDataset(test_set_unpacked, len(classes), transform=preprocessor)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS)
        test_loader_score = DataLoader(test_dataset, batch_size=1,
                                 shuffle=False, num_workers=1)
        return train_loader, test_loader, test_loader_score, len(classes), LABEL_TYPE
    elif LABEL_TYPE == "distance":
        train_dataset = BlockTypeDataset(train_set, None, transform=preprocessor)
        test_dataset = BlockTypeDataset(test_set, None, transform=preprocessor)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS)
        test_loader_score = DataLoader(test_dataset, batch_size=1,
                                       shuffle=False, num_workers=1)
        return train_loader, test_loader, test_loader_score, 1, LABEL_TYPE
    else:
        raise Exception("Unknown label type")