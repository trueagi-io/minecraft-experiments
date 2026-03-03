import torch
import numpy as np
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from train import benchmark
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def preprocess(frame):
    frame = np.resize(np.asarray(frame).astype(np.float32), (320, 240, 3))
    frame = torch.tensor(frame)
    return frame

model_id = "./models/LLaVA-NeXT-Video-7B-hf"

class ModelWrapper:
    def __init__(self):
        self.model = None
        self.processor = None
    def encode(self, x):
        # Load on demand
        if self.model is None:
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True, 
            )
            self.model = self.model.to("cuda")
            self.processor = LlavaNextVideoProcessor.from_pretrained(model_id)
        prompt = "USER: <image>\nASSISTANT:"
        inputs = self.processor(text=prompt, images=[x], padding=True, return_tensors="pt").to(self.model.device)
        #inputs = inputs.to("cuda")
        with torch.no_grad():
            features = torch.stack(self.model.get_image_features(inputs['pixel_values'], image_sizes=inputs['image_sizes']))
        return features

benchmark(
    ModelWrapper(),
    preprocess,
    random_seed=10,
    generalization_set_folder="./2026_LOS_SEGM/Night_clear/Mountain_Range",
    config_path="example_config.json",
    use_precomputed_features=True
)
