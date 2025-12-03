import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleClassifier(nn.Module):
    def __init__(
        self,
        num_features,
        output_size,
        output_activation,
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

        self.output_activation = output_activation

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
        if self.output_activation == "softmax":
            return F.softmax(x, dim=1)
        elif self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.output_activation == "none":
            return x
        else:
            raise ValueError(f"Unknown output activation: {self.output_activation}")
