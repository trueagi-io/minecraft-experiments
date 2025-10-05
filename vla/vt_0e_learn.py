import torch
from torch.utils.data import DataLoader, Dataset
import random
import cv2
import os

from vis_transf import *
from aeblock import *
from utils import *


def train_epoch(ae_model, vt_model, dataloader, optimizer, loss_fn, device):
    from tqdm import tqdm
    #model.train()
    total_loss = 0.0

    for x, target in tqdm(dataloader):
        # x: (B, T, H, W, C),  target: (B, H, W, C)
        x = x.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        # we expect batch size 1 here
        latent = ae_model.encode_straight(x[0])[-1].permute(0, 2, 3, 1)
        latent_p = vt_model(latent.unsqueeze(0))[:, -1]
        pred = ae_model.decode_straight(latent_p.permute(0, 3, 1, 2))[-1]
        # Last frame prediction
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def train_model(ae_model, vt_model, train_dataset, val_dataset, epochs=10, batch_size=8, lr=1e-4, device='cuda'):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    ae_model = ae_model.to(device)
    vt_model = vt_model.to(device)
    params = [p for p in ae_model.parameters()] + [p for p in vt_model.parameters()]
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(ae_model, vt_model, train_loader, optimizer, loss_fn, device)
        #val_loss = evaluate(model, val_loader, loss_fn, device)
        val_loss = 0
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

datafolder = "data/"
savefolder = "result/"
if __name__ == "__main__":
    device = 'cuda'
    vtfile = savefolder + 'vtae_vt.pth'# 'vt_0_ae1.pth' #
    aefile = savefolder + "vtae_ae.pth"# 'ae_1_sparse.pth' #
    video_path = "data/output.mp4"
    frames = extract_frames(video_path, area=320*240, max_frames=10000)

    T = 8
    video_tensor_list = []
    for i in range(len(frames)-T-1):
        video_tensor_list.append(frames[i:i+T+1])
    # train/val split
    random.shuffle(video_tensor_list)
    N = int(0.8 * len(video_tensor_list))
    train_data = VideoPredictDataset(video_tensor_list[:N])
    val_data = VideoPredictDataset(video_tensor_list[N:])
    aes = StackedAE.from_dict(torch.load(aefile, weights_only=True))
    latents = aes.encode_straight(frames[0])[-1]
    (C, H, W) = latents.shape
    fvt = FactorizedVideoTransformer(
        latent_dim=C, H=H, W=W,
        num_frames=T,
        pos_dim=C//4 if C >= 16 else C,
        spatial_depth=3,
        temporal_depth=4,
        n_heads=8 if C//32*32==C else 6,
    )
    fvt.load_state_dict(torch.load(vtfile, weights_only=True))
    #train_model(aes, fvt, train_data, val_data, epochs=10, batch_size=1, lr=2e-6, device=device)
    #torch.save(fvt.state_dict(), savefolder + "vtae_vt.pth")
    #torch.save(aes.to_dict(), savefolder + "vtae_ae.pth")

    aes = aes.to(device)
    fvt = fvt.to(device)
    vid = video_tensor_list[0].to(device)
    with torch.no_grad():
        for i in range(10):
            latents = aes.encode_straight(vid[-T:].to(device))[-1].permute(0, 2, 3, 1)
            latent_p = fvt(latents.unsqueeze(0))[:, -1]
            res = aes.decode_straight(latent_p.permute(0, 3, 1, 2))[-1]
            vid = torch.cat([vid, res])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("predicted.avi", fourcc, 5, (320,240))
    for im in vid:
        real = im.permute(1, 2, 0).cpu().numpy()
        out.write(cv2.cvtColor((real*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        plt.imshow(real)
        plt.show(block=False)
        plt.pause(0.2)
    plt.pause(2.1)
    out.release()
