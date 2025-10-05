import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import *
from aeblock import ConvAEBlock, ConvAEBlockSparse, StackedAE
from training import train_aes_allz

save_folder = "result/"


if __name__ == "__main__":
    video_path = "data/output.mp4"
    frames = extract_frames(video_path, area=320*240, max_frames=10000)
    dataset = VideoFramesDataset(frames)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # The script trains a model together with optimizing z on all layers
    # simultaneously. This allows avoiding layer-by-layer training while
    # keeping the model hierarchical - if we want to train the whole model,
    # we either need to encode straight and then decode straight resulting in
    # unrelated encoders and decoders at each layer but just in one multilayer
    # decoder and encoder - or we need to have own loss for each AE block and
    # cotrain them, because adjacent layers share loss terms.
    # The script can be used to train a model from scratch, although careful
    # selection of learning rates is needed to achieve good results. One
    # can also use checkpoints pretrained layer-by-layer (dense or sparse).
    # Use ae_0_tack_z_test to produce latents since it optimizes z's.
    aes = StackedAE.from_dict(torch.load(save_folder + "ae_0_stack.pth", weights_only=True))
    #ae = StackedAE(
    #    [ConvAEBlockSparse(in_channels=3, num_blocks=6, channels_per_block=12),
    #     ConvAEBlockSparse(in_channels=72, num_blocks=12, channels_per_block=12),
    #     ConvAEBlockSparse(in_channels=144, num_blocks=24, channels_per_block=12),
    #     ConvAEBlockSparse(in_channels=288, num_blocks=64, channels_per_block=12)])
    #ae = StackedAE(
    #    [ConvAEBlock(3, 6), ConvAEBlock(6, 12), ConvAEBlock(12, 24), ConvAEBlock(24, 48)])
    aes = train_aes_allz(aes, frames, epochs=4, lr=1e-4, show_frame_fn=show_frame, lr_z=3e+3) #lr_z=250)
    torch.save(aes.to_dict(), save_folder + "ae_2_allz_stack.pth")
