from train import benchmark
from ultralytics import YOLO
import numpy as np
import torchvision.transforms as T
import torch

def preprocess(frame):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((640, 640))
    ])
    return transform(np.asarray(frame).astype(np.float32)) / 255.0

class Yolo11Wrapper:
    def __init__(self):
        self.model = None
    def encode(self, x):
        # Load on demand
        if self.model is None:
            self.model = YOLO("./models/YOLO11x-cls/yolo11x-cls.pt").model.model[:10].to("cuda")
        with torch.no_grad():
            features = self.model(x)
        return features

if __name__ == "__main__":

    benchmark(
        Yolo11Wrapper(),
        preprocess,
        random_seed=10,
        generalization_set_folder="./2026_LOS_SEGM/Night_clear/Mountain_Range",
        config_path="example_config.json",
        use_precomputed_features=True
    )