import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
from skimage import io
import json
import os
import random
import numpy as np

import sys

from extract_features import extract_dino_features_center_crop, crop_center_features, extract_full_features, base_transform

DATASET_THRESHOLD = 200
TRAIN_NUMBER = int(DATASET_THRESHOLD * 0.25)
TEST_NUMBER = DATASET_THRESHOLD - TRAIN_NUMBER
BATCH_SIZE = 32
NUM_WORKERS = 2
LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_OF_EPOCHS = 10

HIDDEN_LAYER_SIZE = 512

IMG_SIZE_X = 960
IMG_SIZE_Y= 540

TRANSFORM = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((IMG_SIZE_Y, IMG_SIZE_X)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LABEL_TYPE = "class"      # classification
# LABEL_TYPE = "distance"   # regression

class SimpleClassifier(nn.Module):
    def __init__(
        self,
        num_features,
        output_size,
        hidden_size=512,
        num_hidden_layers=1,
        output_activation="softmax"
    ):
        """
        num_features      — input feature size
        output_size       — number of classes OR 1 (regression)
        hidden_size       — size of each hidden layer
        num_hidden_layers — 0, 1, 2, ...
        output_activation — "softmax", "sigmoid", "none" (only these 3 options)
        """
        super().__init__()

        layers = []

        if num_hidden_layers == 0:
            # direct linear mapping
            layers.append(nn.Linear(num_features, output_size))
        else:
            # first layer
            layers.append(nn.Linear(num_features, hidden_size))
            layers.append(nn.ReLU())

            # intermediate hidden layers
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())

            # output layer
            layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)
        self.output_activation = output_activation

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.model(x)

        # activation logic
        if self.output_activation == "softmax":
            return F.softmax(x, dim=1)
        elif self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.output_activation == "none":
            return x
        else:
            raise ValueError(f"Unknown output activation: {self.output_activation}")


# process raw data and create train and test json files.
def make_dataset(directory_path):
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
            if not block_type_LOS in block_types:
                block_types[block_type_LOS] = {}
                block_types[block_type_LOS]["count"] = 1
                block_types[block_type_LOS]["paths"] = []
                block_types[block_type_LOS]["paths"].append(file_name)
            else:
                block_types[block_type_LOS]["count"] += 1
                block_types[block_type_LOS]["paths"].append(file_name)
    for key in block_types:
        if block_types[key]["count"] < DATASET_THRESHOLD:
            continue
        random.shuffle(block_types[key]["paths"])
        train_set[key] = block_types[key]["paths"][:TRAIN_NUMBER]
        test_set[key] = block_types[key]["paths"][TRAIN_NUMBER:TRAIN_NUMBER+TEST_NUMBER]

    json.dump(train_set, open("train_dataset.json",'w'))
    json.dump(test_set, open("test_dataset.json", 'w'))


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

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, f"{self.t_set[idx][1]}.jpg")
        class_idx = self.t_set[idx][0]
        one_hot_vector = np.zeros(self.num_of_classes)
        one_hot_vector[class_idx] = 1
        image = io.imread(img_name)
        if self.transform is not None:
            image = self.transform(image)
        return torch.tensor(one_hot_vector), image


# Just a couple of lines which are being repeated in several functions
def repeatable_code(data, model, classifier):
    labels, images = data
    input_features = model.encode(images.to(device))
    outputs = classifier(input_features)
    return outputs, labels


def get_classes(train_set):
    classes = list(train_set.keys())
    return classes


# compute test loss while training
def test_classifier(model, classifier, test_loader, epoch, criterion, best_loss):
    sum_loss = 0.0
    for i, data in enumerate(test_loader, 0):
        outputs, labels = repeatable_code(data, model, classifier)
        loss = criterion(outputs, labels.to(device))
        sum_loss += loss.item()
    sum_loss /= len(test_loader)
    print(f'[{epoch + 1}] test_loss: {sum_loss}')
    if best_loss > sum_loss:
        torch.save(classifier.state_dict(), "./best_classifier")
        return sum_loss
    else:
        return best_loss


def train_classifier(model, classifier, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    for epoch in range(NUM_OF_EPOCHS):
        running_loss = 0.0
        best_loss = sys.float_info.max
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            outputs, labels = repeatable_code(data, model, classifier)
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] train_loss: {running_loss / 100:.3f}')
                running_loss = 0.0
                classifier.eval()
                best_loss = test_classifier(model, classifier, test_loader, epoch, criterion, best_loss)
                classifier.train()
                torch.save(classifier.state_dict(), "./current_classifier")


# Classifier needs number of features to initialize fully connected layers, so this function
# just returns flattened number of features current encode model outputs for an image
def extract_features_size(model, train_loader):
    for _, data in enumerate(train_loader, 0):
        _, img = data
        img = img[0, :, :, :].unsqueeze(0).to(device)
        features = model.encode(img)
        return features.flatten().size()[0]


# Since we're using model to encode images, I've created wrap class to emulate model.encode() behavior using
# Oleg's code for Dino
class Dinov2Features:
    def encode(img):
        return extract_full_features(img)


# Compute accuracy of the classifier just by summing right answers
def compute_score(model, classifier, test_loader):
    num_of_right_answers = 0
    for i, data in enumerate(test_loader, 0):
        outputs, labels = repeatable_code(data, model, classifier)
        predicted_class_label = torch.argmax(outputs)
        real_class_label = torch.argmax(labels)
        if predicted_class_label.item() == real_class_label.item():
            num_of_right_answers += 1
    print(f"Accuracy: {float(num_of_right_answers) * 100 / len(test_loader)}%")


def main():
    # make_dataset("./2025_LOS/") # need to be launched on raw data to generate json files
    train_set = read_dataset("train_dataset.json")
    test_set = read_dataset("test_dataset.json")
    classes = get_classes(train_set)
    train_set_unpacked = unpack_dataset(train_set, classes)
    test_set_unpacked = unpack_dataset(test_set, classes)
    train_dataset = BlockTypeDataset(train_set_unpacked, len(classes), transform=base_transform)
    test_dataset = BlockTypeDataset(test_set_unpacked, len(classes), transform=base_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)

    model = Dinov2Features  # this could be replaced with any other model which has .encode(img) function

    classifier = SimpleClassifier(len(classes), extract_features_size(model, train_loader)).to(device)
    train_classifier(model, classifier, train_loader, test_loader)
    compute_score(model, classifier, DataLoader(test_dataset, batch_size=1,
                                                shuffle=False, num_workers=NUM_WORKERS))


if __name__ == '__main__':
    main()
