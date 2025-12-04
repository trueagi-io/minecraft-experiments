from manager import DatasetManager
from config import LabelType, PATH_TO_RAW_DATA


if __name__ == "__main__":
    manager = DatasetManager(
        directory=PATH_TO_RAW_DATA,
        label_type=LabelType.DISTANCE,
        dataset_size=200,
        train_split=0.25,
        num_bins=50
    )

    train_file, test_file, t = manager.create("los_dataset")

    print("Generation time:", t)
