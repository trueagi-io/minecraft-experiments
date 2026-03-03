from transformers import AutoProcessor, AutoModel
import torch
import numpy as np
from train import benchmark

def preprocess(frame):
    frame = np.resize(np.asarray(frame).astype(np.float32), (504, 504, 3))
    frame = torch.tensor(frame)
    return frame

model_id = "./models/Eagle2.5-8B"

class ModelWrapper:
    def __init__(self):
        self.model = None
        self.processor = None
    def encode(self, x):
        # Load on demand
        if self.model is None:
            self.model = AutoModel.from_pretrained(model_id,
                    trust_remote_code=True, torch_dtype=torch.bfloat16)
                    #attn_implementation="sdpa")
            self.model = self.model.to("cuda")
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
            self.processor.tokenizer.padding_side = "left"
        text_list = ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image-1><|im_end|>\n<|im_start|>assistant\n']
        inputs = self.processor(text=text_list, images=[x], return_tensors="pt", padding=True)
        with torch.no_grad():
            features = self.model.vision_model(inputs['pixel_values']).last_hidden_state
        return features


benchmark(
    ModelWrapper(),
    preprocess,
    random_seed=10,
    generalization_set_folder="./2026_LOS_SEGM/Night_clear/Mountain_Range",
    config_path="example_config.json",
    use_precomputed_features=False
)
