import cv2
import torch
import numpy as np
from PIL import Image
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


# =====================================================================
# Function 1: Extract DINO features from CENTRAL CROP of image
# =====================================================================
def extract_dino_features_center_crop(frame, crop_ratio=0.5):

    h, w = frame.shape[2:]

    new_h = int(h * crop_ratio)
    new_w = int(w * crop_ratio)

    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2
    y2 = y1 + new_h
    x2 = x1 + new_w

    # Crop and prepare
    crop_frame = frame[:, :, y1:y2, x1:x2]
    return crop_frame


# =====================================================================
# Function 2: Crop CENTER of feature-space grid (after DINO processing)
# =====================================================================
def crop_center_features(features, crop_ratio=0.5):
    """
    Crop central patch region from feature tokens: [1, N, D].
    """
    B, N, D = features.shape
    h = w = int(np.sqrt(N))
    assert h * w == N, "Features do not form a square grid."

    new_h = int(h * crop_ratio)
    new_w = int(w * crop_ratio)

    top = (h - new_h) // 2
    left = (w - new_w) // 2
    bottom = top + new_h
    right = left + new_w

    feats_2d = features.reshape(1, h, w, D)
    cropped = feats_2d[:, top:bottom, left:right, :]
    cropped = cropped.reshape(1, -1, D)

    return cropped

def extract_full_features(img):
    with torch.no_grad():
        full_feats = dinov2.get_intermediate_layers(img, n=1)[0]
        return full_feats

class Dinov2Features_full:
    def encode(self, img):
        return extract_full_features(img)

class Dinov2Features_crop_features:
    def __init__(self, crop_ratio=0.5):
        self.crop_ratio = crop_ratio

    def encode(self, img):
        features = extract_full_features(img)
        return crop_center_features(features, self.crop_ratio)

class Dinov2Features_crop_image:
    def __init__(self, crop_ratio=0.5):
        self.crop_ratio = crop_ratio

    def encode(self, img):
        img = extract_dino_features_center_crop(img, self.crop_ratio)
        return extract_full_features(img)