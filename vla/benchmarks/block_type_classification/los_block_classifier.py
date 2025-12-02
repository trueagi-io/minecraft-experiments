import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys

from Dataset import CreateDataloader

LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_OF_EPOCHS = 10

OUTPUT_ACTIVATION = "none" # "softmax", "sigmoid", "none" (only these 3 options)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CRITERION = nn.MSELoss()
# CRITERION = nn.CrossEntropyLoss()

class SimpleClassifier(nn.Module):
    def __init__(
        self,
        num_features,
        output_size,
        hidden_size=512,
        num_hidden_layers=1
    ):
        """
        num_features      — input feature size
        output_size       — number of classes OR 1 (regression)
        hidden_size       — size of each hidden layer
        num_hidden_layers — 0, 1, 2, ...
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

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.model(x)

        # activation logic
        if OUTPUT_ACTIVATION == "softmax":
            return F.softmax(x, dim=1)
        elif OUTPUT_ACTIVATION == "sigmoid":
            return torch.sigmoid(x)
        elif OUTPUT_ACTIVATION == "none":
            return x
        else:
            raise ValueError(f"Unknown output activation: {OUTPUT_ACTIVATION}")

# Just a couple of lines which are being repeated in several functions
def repeatable_code(data, model, classifier):
    labels, images = data
    input_features = model.encode(images.to(device))
    outputs = classifier(input_features)
    return outputs, labels.to(device)

# compute test loss while training
def test_classifier(model, classifier, test_loader, epoch, best_loss):
    sum_loss = 0.0
    for i, data in enumerate(test_loader, 0):
        outputs, labels = repeatable_code(data, model, classifier)
        loss = CRITERION(outputs, labels.to(device))
        sum_loss += loss.item()
    sum_loss /= len(test_loader)
    print(f'[{epoch + 1}] test_loss: {sum_loss}')
    if best_loss > sum_loss:
        torch.save(classifier.state_dict(), "./best_classifier")
        return sum_loss
    else:
        return best_loss

def train_classifier(model, classifier, train_loader, test_loader):
    optimizer = optim.SGD(classifier.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    for epoch in range(NUM_OF_EPOCHS):
        running_loss = 0.0
        best_loss = sys.float_info.max
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            outputs, labels = repeatable_code(data, model, classifier)
            loss = CRITERION(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] train_loss: {running_loss / 100:.3f}')
                running_loss = 0.0
                classifier.eval()
                best_loss = test_classifier(model, classifier, test_loader, epoch, best_loss)
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

# Compute accuracy of the classifier just by summing right answers
def compute_score_classifier(model, classifier, test_loader):
    num_of_right_answers = 0
    for i, data in enumerate(test_loader, 0):
        outputs, labels = repeatable_code(data, model, classifier)
        predicted_class_label = torch.argmax(outputs)
        real_class_label = torch.argmax(labels)
        if predicted_class_label.item() == real_class_label.item():
            num_of_right_answers += 1
    print(f"Accuracy: {float(num_of_right_answers) * 100 / float(len(test_loader))}%")

def compute_score_regression(model, classifier, test_loader):
    sum_mse = 0
    mse_loss = nn.MSELoss()
    for i, data in enumerate(test_loader, 0):
        outputs, labels = repeatable_code(data, model, classifier)
        sum_mse += mse_loss(outputs, labels).item()
    print(f"Average MSE: {float(sum_mse) / float(len(test_loader))}")

def benchmark(model, preprocessor, train_set_json_file, test_set_json_file):
    train_loader, test_loader, test_loader_score, num_classes, label_type = CreateDataloader(train_set_json_file, test_set_json_file, preprocessor)
    classifier = SimpleClassifier(extract_features_size(model, train_loader), num_classes).to(device)
    train_classifier(model, classifier, train_loader, test_loader)
    if label_type == "type":
        compute_score_classifier(model, classifier, test_loader_score)
    else:
        compute_score_regression(model, classifier, test_loader_score)