from manager import DatasetManager
from config import LabelType


if __name__ == "__main__":
    manager = DatasetManager(
        directory="./2025_LOS/Day_clear/",
        label_type=LabelType.DISTANCE,
        dataset_size=200,
        train_split=0.25,
        num_bins=50
    )

    train_file, test_file = manager.create("los_dataset")