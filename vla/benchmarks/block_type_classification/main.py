# from los_block_classifier import benchmark
from DinoV2Features import Dinov2Features_crop_image, preprocessor
from Dataset_old import make_dataset
from train import benchmark
from config import LabelType

# make_dataset("./2025_LOS/", "regression_dataset") # need to be launched on raw data to generate json files
# benchmark(Dinov2Features_crop_image(), preprocessor, 'train_los_dataset.json', 'test_los_dataset.json', label_type="distance")

if __name__ == "__main__":
    benchmark(
        Dinov2Features_crop_image(),
        preprocessor,
        'train_los_dataset.json',
        'test_los_dataset.json',
        label_type=LabelType.DISTANCE
    )
