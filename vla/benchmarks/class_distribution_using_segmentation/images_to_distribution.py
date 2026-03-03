# This file is used to convert segmented images to npy files which contains class distribution
import numpy as np
from pathlib import Path
from PIL import Image

paths = []
unique_colors_list = []
basepath = Path("./2026_LOS_SEGM")
segmentation_subfolders = list(basepath.rglob("segmentation"))

stub = 0

for segm_path in segmentation_subfolders:
    images_folder = list(segm_path.rglob("*.png"))
    for segm_image_path in images_folder:
        segm_img = np.array(Image.open(segm_image_path))
        unique = list(np.unique(segm_img.reshape(-1, segm_img.shape[2]), axis=0))
        for unique_item in unique:
            unique_item = list(unique_item)
            if unique_item not in unique_colors_list:
                unique_colors_list.append(unique_item)

unique_colors_dict = {str(value): index for index, value in enumerate(unique_colors_list)}
samples = []
num_classes = len(unique_colors_dict)
for segm_path in segmentation_subfolders:
    distribution_path = segm_path.parent / 'distributions'
    distribution_path.mkdir(parents=True, exist_ok=True)
    images_folder = list(segm_path.rglob("*.png"))
    for segm_image_path in images_folder:
        segm_img = np.array(Image.open(segm_image_path))
        class_probs = [0] * num_classes
        color_list = list(segm_img.reshape(-1, segm_img.shape[2]))
        for color in color_list:
            class_probs[unique_colors_dict[str(list(color))]] += 1
        class_probs = np.array(class_probs) / len(color_list)
        np.save(f'{str(distribution_path)}/{segm_image_path.stem}.npy', class_probs)
