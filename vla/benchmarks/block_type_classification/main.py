from los_block_classifier import benchmark
from DinoV2Features import Dinov2Features_crop_image, preprocessor
from Dataset import make_dataset

# make_dataset("./2025_LOS/", "regression_dataset") # need to be launched on raw data to generate json files
benchmark(Dinov2Features_crop_image(), preprocessor, 'train_regression_dataset.json', 'test_regression_dataset.json')