from train import benchmark

import torch
from transformers import AutoModel, AutoProcessor

ckpt = "./models/siglip2-so400m-patch14-384"

processor = AutoProcessor.from_pretrained(ckpt)

def preprocess(frame):
    inputs = processor(images=[frame], return_tensors="pt").to("cuda")
    return inputs.data["pixel_values"].squeeze()

class SiglipWrapper:
    def __init__(self):
        self.model = None
    def encode(self, x):
        # Load on demand
        if self.model is None:
            self.model = AutoModel.from_pretrained(ckpt, device_map="auto").cuda().eval()
        with torch.no_grad():
            features = self.model.get_image_features(x)
        return features


if __name__ == "__main__":

    benchmark(
        SiglipWrapper(),
        preprocess,
        random_seed=10,
        generalization_set_folder="./2026_LOS_SEGM/Night_clear/Mountain_Range",
        config_path="example_config.json",
        use_precomputed_features=True
    )