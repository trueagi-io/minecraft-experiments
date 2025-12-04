from enum import Enum


class LabelType(str, Enum):
    DISTANCE = "distance"
    TYPE_CLASSIFICATION = "type"


PATH_TO_RAW_DATA = "/path/to/folder"

FEATURE_STORE_PATH = "/path/to/folder"


class TrainConfig:
    learning_rate = 1e-3
    momentum = 0.9
    epochs = 10
    output_activation = "none"  # "none", "softmax", "sigmoid"
