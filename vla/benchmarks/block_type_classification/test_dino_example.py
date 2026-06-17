from DinoV2Features import Dinov2Features_crop_image, preprocessor
from train import benchmark
from config import LabelType


if __name__ == "__main__":
    benchmark(
        Dinov2Features_crop_image(),
        preprocessor,
        label_type=LabelType.DISTANCE,
        random_seed=10,
        config_path="example_config.json"
    )
