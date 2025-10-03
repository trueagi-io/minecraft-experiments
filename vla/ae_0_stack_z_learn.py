import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from aeblock import ConvAEBlock, StackedAE
from training import train_aes_top

save_folder = "result/"

if __name__ == "__main__":
    video_path = "data/output.mp4"
    frames = extract_frames(video_path, area=320*240, max_frames=10000)
    dataset = VideoFramesDataset(frames)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Training procedure:
    # Stacked AE is trained layer-by-layer.
    # For each new layer, decoder_only=True for training the decoder
    # via optimizing `z` using SGD is used first. Then, decoder_only=False
    # and add_layer=False should be used to train the encoder. Optionally,
    # the process can be repeated multiple times.
    # At first, no checkpoint is available (checkpoint = None).
    # Then, the same file can be used to store checkpoints with more and more
    # layers (add_layer=True).
    checkpoint = save_folder + "ae_0_stack.pth" # None
    add_layer = False
    decoder_only = False
    if checkpoint is None:
        ae_model = StackedAE([ConvAEBlock(3, 6, 6, 2, 2)])
    else:
        ae_model = StackedAE.from_dict(torch.load(checkpoint, weights_only=True))
        if add_layer:
            e = list(ae_model.aes[-1].encoder.children())[0]
            layer = ConvAEBlock(e.out_channels, e.out_channels*2, 6, 2, 2)
            ae_model.aes.append(layer)
    ae_model = train_aes_top(ae_model, dataloader, decoder_only, lr=1e-3, show_frame_fn=show_frame)
    torch.save(ae_model.to_dict(), save_folder + "ae_0_stack.pth")

