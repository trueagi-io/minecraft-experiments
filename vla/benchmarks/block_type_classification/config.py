from enum import Enum


class LabelType(str, Enum):
    DISTANCE = "distance"
    TYPE_CLASSIFICATION = "type"


PATH_TO_RAW_DATA = "../../../Mountain_Range"

FEATURE_STORE_PATH = "./precomputed_features"


class TrainConfig:
    learning_rate = 1e-3
    momentum = 0.9
    epochs = 10
    output_activation = "none"  # "none", "softmax", "sigmoid"
