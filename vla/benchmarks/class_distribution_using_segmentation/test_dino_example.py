from DinoV2Features import Dinov2Features_full, preprocessor
from train import benchmark


if __name__ == "__main__":

    benchmark(
        Dinov2Features_full(),
        preprocessor,
        random_seed=10,
        config_path="example_config.json",
        use_precomputed_features=True
    )
