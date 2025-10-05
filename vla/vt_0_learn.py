import torch
from torch.utils.data import DataLoader, Dataset
import random
import cv2
import os

from vis_transf import *
from aeblock import *
from utils import *

datafolder = "data/"
savefolder = "result/"
if __name__ == "__main__":
    device = 'cuda'
    sources = {
        'ae0': {'latents': datafolder + "ae_0_latents.pth",
                'vt': savefolder + 'vt_0_ae0.pth',
                'ae': savefolder + "ae_0_stack.pth"},
        'ae1': {'latents': datafolder + "ae_1_latents.pth",
                'vt': savefolder + 'vt_0_ae1.pth',
                'ae': savefolder + 'ae_1_sparse.pth'},
    }
    s = 'ae1'
    latent_dict = torch.load(sources[s]['latents'], weights_only=True)
    latents = []
    T = 8
    for i in range(len(latent_dict.keys())):
        latents.append(latent_dict[i].permute(1, 2, 0))
    video_tensor_list = []
    for i in range(len(latents)-T-1):
        video_tensor_list.append(torch.stack(latents[i:i+T+1]))
    # train/val split
    random.shuffle(video_tensor_list)
    N = int(0.8 * len(video_tensor_list))
    train_data = VideoPredictDataset(video_tensor_list[:N])
    val_data = VideoPredictDataset(video_tensor_list[N:])
    (H, W, C) = latents[0].shape
    fvt = FactorizedVideoTransformer(
        latent_dim=C, H=H, W=W,
        num_frames=T,
        pos_dim=C//4 if C >= 16 else C,
        spatial_depth=3,
        temporal_depth=4,
        n_heads=8 if C//32*32==C else 6,
    )
    if os.path.exists(sources[s]['vt']):
        fvt.load_state_dict(torch.load(sources[s]['vt'], weights_only=True))
    train_model(fvt, train_data, val_data, epochs=30, batch_size=1, lr=1e-4, device=device)
    torch.save(fvt.state_dict(), sources[s]['vt'])

    aes = StackedAE.from_dict(torch.load(sources[s]['ae'], weights_only=True)).to(device)
    fvt = fvt.to(device)
    vid = video_tensor_list[0].to(device)
    with torch.no_grad():
        for i in range(10):
            res = fvt(vid[-T:].unsqueeze(0))
            vid = torch.cat([vid, res[0][-1:]])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("predicted.avi", fourcc, 5, (320,240))
    for lat in vid:
        im = aes.decode_straight(lat.permute(2, 0, 1))[-1].detach()
        real = im.permute(1, 2, 0).cpu().numpy()
        out.write(cv2.cvtColor((real*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        plt.imshow(real)
        plt.show(block=False)
        plt.pause(0.2)
    plt.pause(2.1)
    out.release()
