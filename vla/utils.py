import torch
from torch.utils.data import Dataset
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')


def extract_frames(video_path, area=320*240, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        coef = math.sqrt(frame.shape[0]*frame.shape[1]/area)
        resize = (int(frame.shape[1]/coef), int(frame.shape[0]/coef))
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1

    cap.release()
    frames = np.array(frames).astype(np.float32) / 255.0
    print(frames.shape)
    frames = torch.tensor(frames).permute(0, 3, 1, 2)  # (N, C, H, W)
    return frames


class VideoFramesDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]


class VideoClipDataset(Dataset):
    def __init__(self, frames, N_seq=8, N_shift=1):
        """
        frames: Tensor (T, C, H, W)
        """
        super().__init__()
        self.frames = frames  # (T, C, H, W)
        self.N_seq = N_seq
        self.N_shift = N_shift
        self.num_sequences = max(0, (self.frames.shape[0] - N_seq) // N_shift + 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.N_shift
        end = start + self.N_seq
        # Choose a slice and permute it
        # From (T_seq, C, H, W) → (C, T_seq, H, W)
        clip = self.frames[start:end].permute(1, 0, 2, 3)
        return clip


def show_frame(real, fake):
    plt.close()
    # if len(real.shape) > 3 ...
    real = real.permute(1, 2, 0).detach().cpu().numpy()
    fake = fake.permute(1, 2, 0).detach().cpu().numpy()
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(real)
    axs[0].set_title("Original")
    axs[1].imshow(fake)
    axs[1].set_title("Generated")
    #plt.ion()
    plt.show(block=False)
    plt.pause(0.1)

