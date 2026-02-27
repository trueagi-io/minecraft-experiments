import cv2
import torch
import numpy as np
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# Load DINOv2
# =====================================================================
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(device)

# =====================================================================
# Preprocessing
# =====================================================================
preprocessor = T.Compose([
    T.ToTensor(),
    T.Resize((504, 504)),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def extract_full_features(img):
    with torch.no_grad():
        full_feats = dinov2.get_intermediate_layers(img, n=1)[0]
        return full_feats

class Dinov2Features_full:
    def encode(self, img):
        return extract_full_features(img)