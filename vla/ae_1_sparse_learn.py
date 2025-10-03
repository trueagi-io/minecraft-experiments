import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from aeblock import ConvAEBlockSparse, StackedAE
from training import train_aes_top

save_folder = "result/"

if __name__ == "__main__":
    num_feat = [(6, 12), (12, 12), (24, 12), (64, 12)] # 64->48
    video_path = "data/output.mp4"
    frames = extract_frames(video_path, area=320*240, max_frames=10000)
    dataset = VideoFramesDataset(frames)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # This is the same script as ae_0_stack, but with different type
    # of autoencoder blocks. Sparse blocks construct more features, but
    # try to make them sparse by block-wise softmax.
    # Optimizing z by setting decoder_only=True may not be a good approach
    # here, because optimizaton of z doesn't take constraints into account,
    # that is, the latent code will simply be very redundant - not compressive.
    # The idea of having sparse latent code is to make AE large, so it
    # could memorize abstracted image fragments.
    # Consequently, optimization of all z is not performed atm, so AE could
    # rather be trained end-to-end rather than independently for each layer.
    # However, we keep layer-by-layer training.

    checkpoint = save_folder + "ae_1_sparse.pth" # None
    add_layer = False
    decoder_only = False
    if checkpoint is None:
        ae_model = StackedAE([ConvAEBlockSparse(3, num_feat[0][0], num_feat[0][1], 6, 2, 2)])
    else:
        ae_model = StackedAE.from_dict(torch.load(checkpoint, weights_only=True))
        if add_layer:
            n = len(ae_model.aes)
            o = ae_model.aes[-1].total_out_channels
            layer = ConvAEBlockSparse(o, num_feat[n][0], num_feat[n][1], 6, 2, 2)
            ae_model.aes.append(layer)
    ae_model = train_aes_top(ae_model, dataloader, decoder_only, lr=1e-3)
    torch.save(ae_model.to_dict(), save_folder + "ae_1_sparse.pth")

