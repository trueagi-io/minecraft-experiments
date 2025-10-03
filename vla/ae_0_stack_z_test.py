import torch
import torch.nn as nn
from utils import *
from aeblock import StackedAE

def test_model(frames, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data = VideoFramesDataset(frames)
    total_loss = 0.0
    total_tloss = 0.0
    model = model.to(device)
    all_topz = []
    for i, frame in enumerate(data):
        frame = frame.to(device)
        zs, tloss = model.optimize_latents(frame, lr=4e+3)
        loss = nn.MSELoss()(model.aes[0].decoder(zs[0]), frame)
        all_topz.append(zs[-1].to('cpu'))
        #print(i, loss.item(), tloss.item())
        #x_rec = model(frame)
        #x_rec = model.decode_straight(z=zs_opt[-1])[-1]
        #loss = loss_fn(x_rec, frame)
        #show_frame(frame, x_rec) #.to('cpu'))
        total_loss += loss.item()
        total_tloss += tloss.item()
    print(f"Loss: {total_loss / len(data):.6f} | TLoss: {total_tloss / len(data):.6f}")
    return all_topz


if __name__ == "__main__":
    aes = StackedAE.from_dict(torch.load("result/ae_0_stack.pth", weights_only=True))
    frames = extract_frames("data/output.mp4", area=320*240, max_frames=10000)
    latents = test_model(frames, aes)
    latent_dict = {idx: z for idx, z in enumerate(latents)}
    torch.save(latent_dict, "data/ae_0_latents.pth")


