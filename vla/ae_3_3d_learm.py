import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import *

from aeblock import *
from training import *

def show_vframe(real, fake):
    show_frame(real.permute(1, 0, 2, 3)[-1], fake.permute(1, 0, 2, 3)[-1])
    #show_frame(fake.permute(1, 0, 2, 3)[-14-4], fake.permute(1, 0, 2, 3)[-12-4])

save_folder = "result/"

if __name__ == "__main__":
    video_path = "data/output.mp4"
    frames = extract_frames(video_path, area=320*240, max_frames=10000)
    dataset = VideoClipDataset(frames, 96, 6)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    checkpoint = save_folder + "ae_3_3d.pth" # None
    add_layer = False
    decoder_only = False
    if checkpoint is None:
        #ae_model = StackedAE([Conv3DAEBlock(3, 12, 6, 2, 2)])
        # No compression over T, padding to avoid side effects for deconvolution
        ae_model = StackedAE([Conv3DAEBlock(3, 6, kernel_size=(5,6,6), stride=(1,2,2), padding=(2,2,2))])
    else:
        ae_model = StackedAE.from_dict(torch.load(checkpoint, weights_only=True))
        if add_layer:
            e = list(ae_model.aes[-1].encoder.children())[0]
            #layer = Conv3DAEBlock(e.out_channels, e.out_channels*4, 6, 2, 2)
            # No compression over T
            layer = Conv3DAEBlock(e.out_channels, e.out_channels*2, kernel_size=(5,6,6), stride=(1,2,2), padding=(2,2,2))
            ae_model.aes.append(layer)
    ae_model = train_aes_top(ae_model, dataloader, decoder_only, lr=1e-5, show_frame_fn=show_vframe)
    torch.save(ae_model.to_dict(), save_folder + "ae_3_3d.pth")
