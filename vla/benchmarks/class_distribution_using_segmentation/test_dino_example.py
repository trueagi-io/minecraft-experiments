from DinoV2Features import Dinov2Features_full, preprocessor
from train import benchmark
from config import construct_configs


if __name__ == "__main__":

    benchmark(
        Dinov2Features_full(),
        preprocessor,
        random_seed=10,
        generalization_set_folder="./2026_LOS_SEGM/Night_clear/Mountain_Range",
        config_path="example_config.json",
        use_precomputed_features=True
    )
