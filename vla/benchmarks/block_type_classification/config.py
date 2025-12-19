from enum import Enum
import warnings

class LabelType(str, Enum):
    DISTANCE = "distance"
    TYPE_CLASSIFICATION = "type"

class ConfigPaths:
    path_to_raw_data = None
    feature_store_path = "./precomputed"

class TrainConfig:
    learning_rate = 1e-3
    momentum = 0.9
    epochs = 10
    output_activation = "none"  # "none", "softmax", "sigmoid"
    num_layers = 1
    layers_sizes = [512]

def construct_configs(path_to_raw_data=None, feature_store_path=None,
        learning_rate=1e-3, momentum=0.9, epochs=10, output_activation="none", num_layers=1, layers_sizes=[512]):
    TrainConfig.learning_rate = learning_rate
    TrainConfig.momentum = momentum
    TrainConfig.epochs = epochs
    TrainConfig.output_activation = output_activation
    TrainConfig.num_layers = num_layers
    TrainConfig.layers_sizes = layers_sizes
    if not path_to_raw_data:
        raise ValueError("Path to raw data must be set")
    else:
        ConfigPaths.path_to_raw_data = path_to_raw_data
    if not feature_store_path:
        warnings.warn("Feature store path is not set by config. If benchmark will be ran with use_precomputed_features "
                      "default folder ./precomputed will be used to store features")
    else:
        ConfigPaths.feature_store_path = feature_store_path