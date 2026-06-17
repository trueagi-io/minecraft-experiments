import json
from manager import DatasetManager
from config import LabelType, ConfigPaths, construct_configs


if __name__ == "__main__":
    with open("example_config.json", 'r') as f:
        config_dict = json.load(f)
    construct_configs(**config_dict)

    manager = DatasetManager(
        directory=ConfigPaths.path_to_raw_data,
        label_type=LabelType.DISTANCE,
        dataset_size=200,
        train_split=0.25,
        num_bins=50
    )

    train_file, test_file = manager.create("los_dataset")