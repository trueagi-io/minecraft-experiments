from transformers import AutoProcessor, AutoModel
import torch
import numpy as np
import cv2
from benchmarks.block_type_classification import benchmark, LabelType

def preprocess(frame):
    # it seems that just turning into tensor is enough here
    # preprocessing itself is done by the model processor
    # however, central crop could be performed as an option
    frame = np.asarray(frame)
    #frame = cv2.resize(frame, (240, 320))
    #frame = frame[frame.shape[0]//2-120:frame.shape[0]//2+120,
    #              frame.shape[1]//2-160:frame.shape[1]//2+160]
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.array(frame).astype(np.float32) # / 255.0
    frame = torch.tensor(frame) #.permute(2, 0, 1)  # (C, H, W)
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
        #inputs = inputs.to("cuda")
        with torch.no_grad():
            features = self.model.vision_model(inputs['pixel_values']).last_hidden_state
        return features


benchmark(
    ModelWrapper(),
    preprocess,
    label_type=LabelType.DISTANCE, #LabelType.TYPE_CLASSIFICATION, #
    random_seed=10,
    generalization_set_folder="./2025_LOS/Night_clear/Mountain_Range",
    config_path="example_config.json"
)
