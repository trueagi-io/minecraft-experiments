import torch
import torch.nn as nn
from utils import *
from aeblock import StackedAE

@torch.no_grad
def collect_latents(frames, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    all_topz = []
    total_loss = 0
    for frame in frames:
        frame = frame.to(device)
        z = model.encode_straight(frame)[-1]
        recon = model.decode_straight(z)[-1]
        #show_frame(frame, recon)
        total_loss += nn.MSELoss()(frame, recon).item()
        all_topz.append(z.to('cpu'))
    print(f"Loss: {total_loss / len(frames):.6f}")
    return all_topz


if __name__ == "__main__":

    frames = extract_frames("data/output.mp4", area=320*240, max_frames=10000)
    model = StackedAE.from_dict(torch.load("result/ae_1_sparse.pth", weights_only=True))
    latents = collect_latents(frames, model)
    latent_dict = {idx: z for idx, z in enumerate(latents)}
    torch.save(latent_dict, "data/ae_1_latents.pth")
