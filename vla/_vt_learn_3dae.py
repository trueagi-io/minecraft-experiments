import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from aeblock import StackedAE
import random
from tqdm import tqdm
from utils import *


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    #model.train()
    total_loss = 0.0

    for x, target in tqdm(dataloader):
        # x: (B, T, H, W, C),  target: (B, H, W, C)
        x = x.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        out = model(x)  # (B, T, H, W, C)
        # Возьмём предсказание последнего кадра
        pred = out[:, -1]  # (B, H, W, C)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    #model.eval()
    total_loss = 0.0

    for x, target in dataloader:
        x = x.to(device)
        target = target.to(device)
        out = model(x)
        pred = out[:, -1]
        loss = loss_fn(pred, target)
        total_loss += loss.item()

    return total_loss / len(dataloader)

def train_model(model, train_dataset, val_dataset, epochs=10, batch_size=8, lr=1e-4, device='cuda'):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")


class VideoLatentsDataset(Dataset):
    def __init__(self, videos):
        """
        videos: список видеороликов, каждый shape: (T+1, H, W, C)
        """
        self.videos = videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        clip = self.videos[idx]
        x = clip[:-1]  # T кадров — вход
        y = clip[-1]   # последний — цель (target)
        return x, y


if __name__ == "__main__":
    #ae_model = StackedAE.from_dict(torch.load("ae3d_6a4.pth", weights_only=True))
    ae_model = StackedAE.from_dict(torch.load("ae3dc_a4.pth", weights_only=True))
    frames = extract_frames("output.mp4", area=320*240, max_frames=10000)
    dataset = VideoClipDataset(frames, 24, 6) #192
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    ae_model = ae_model.to('cuda')
    video_tensor_list = []
    T = 6
    with torch.no_grad():
        for clip in dataloader:
            clip = clip.to('cuda')
            z = ae_model.encode_straight(clip)[-1][0].permute(1, 2, 3, 0)
            for i in range(z.size(0)-T-1):
                video_tensor_list.append(z[i:i+T+1].to('cpu'))#clone())
    # train/val split
    random.shuffle(video_tensor_list)
    N = int(0.8 * len(video_tensor_list))
    train_data = VideoLatentsDataset(video_tensor_list[:N])
    val_data = VideoLatentsDataset(video_tensor_list[N:])

    fvt = FactorizedVideoTransformer(
        latent_dim=256,#768,
        H=15,
        W=20,
        num_frames=T,
        pos_dim=144*2,
        spatial_depth=3,
        temporal_depth=4,
        n_heads=8,
    )
    fvt.load_state_dict(torch.load("vis_tran_3d1c.pth", weights_only=True))
    # запуск обучения
    train_model(fvt, train_data, val_data, epochs=30, batch_size=1, lr=3e-5, device='cuda')
    torch.save(fvt.state_dict(), "vis_tran_3d1c.pth")

