from DinoV2Features import Dinov2Features_crop_image, preprocessor
from train import benchmark
from config import LabelType


if __name__ == "__main__":
    benchmark(
        Dinov2Features_crop_image(),
        preprocessor,
        'train_los_dataset.json',
        'test_los_dataset.json',
        label_type=LabelType.DISTANCE
    )
