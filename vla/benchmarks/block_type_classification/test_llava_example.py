import torch
import numpy as np
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from benchmarks.block_type_classification import benchmark, LabelType

def preprocess(frame):
    frame = np.asarray(frame)
    frame = np.array(frame).astype(np.float32)
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
            features = self.model.get_image_features(inputs['pixel_values'], image_sizes=inputs['image_sizes'])[0]
        return features

benchmark(
    ModelWrapper(),
    preprocess,
    label_type=LabelType.TYPE_CLASSIFICATION, #LabelType.DISTANCE, 
    random_seed=10,
    generalization_set_folder="./2025_LOS/Night_clear/Mountain_Range",
    config_path="example_config.json"
)
 