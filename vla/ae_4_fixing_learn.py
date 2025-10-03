import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

from utils import *
from aeblock import *


def train_model(dataloader, aes, epoches=30, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    aes = aes.to(device)

    main_optimizers = [torch.optim.Adam(ae.ae_main.parameters(), lr=lr) for ae in aes.aes]
        #torch.optim.Adam([param for ae in aes.aes for param in ae.ae_main.parameters()], lr=lr)
    fixing_optimizers = [torch.optim.Adam(ae.ae_fix.get_encoder_params(), lr=lr) for ae in aes.aes]
    loss_fn = nn.MSELoss()
    best_loss = 1e+10

    for epoch in range(epoches):
        total_loss1 = 0.0
        total_loss2 = 0.0

        for i, frame in enumerate(tqdm(dataloader)):
            x = frame.to(device)
            for n in range(len(aes.aes)):
                z = aes.aes[n].ae_main.encode(x)
                r = aes.aes[n].ae_main.decode(z)
                loss = loss_fn(x, r)
                main_optimizers[n].zero_grad()
                loss.backward()
                main_optimizers[n].step()
                total_loss1 += loss.item()
                x_ = z
                for inner in range(5):
                    dx = (x - r).detach()
                    dz = aes.aes[n].ae_fix.encode(dx)
                    r2 = aes.aes[n].ae_main.decode(z.detach()+dz)
                    loss2 = loss_fn(x, r2)
                    fixing_optimizers[n].zero_grad()
                    loss2.backward()
                    fixing_optimizers[n].step()
                    if loss2 > loss:
                        break
                    else:
                        loss = loss2
                        z = z + dz
                        r = r2
                x = x_.detach() #   z.detach() #
                total_loss2 += loss2.item() #loss?
            #if i == len(frames) - 1 and epoch % 10 == 0:
            #    show_frame(frames[i], r2[0])

        #if total_loss < best_loss:
        #    best_loss = total_loss
        #    torch.save(ae.to('cpu').to_dict(), "aest_stack3block.pth")
        #    ae.to('cuda')

        print(f"[Epoch {epoch + 1}/{epoches}] Loss: {total_loss1 / len(dataloader):.6f} {total_loss2 / len(dataloader):.6f}")

    return aes.to('cpu')


save_folder = "result/"

if __name__ == "__main__":
    video_path = "data/output.mp4"
    frames = extract_frames(video_path, area=320*240, max_frames=10000)
    dataset = VideoFramesDataset(frames)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # PoC script, which replaces SGD z optimization with additional
    # fixing encoder. It works in general, at least for one layer,
    # but training procedure requires refinement - how main AE and
    # fixing encoders are updated especially in hierarchical mode,
    # when all z should be optimized simultaneously - right now
    # it is done layer-by-layer.
    checkpoint = None # save_folder + "ae_4_fix.pth"
    if checkpoint is None:
        ae_model = StackedAE([
            FixingAEBlock(ConvAEBlock(3, 6), ConvAEBlock(3, 6)),
            #FixingAEBlock(ConvAEBlock(6, 12), ConvAEBlock(6, 12))
        ])
    else:
        ae_model = StackedAE.from_dict(torch.load(checkpoint, weights_only=True))
    ae_model = train_model(dataloader, ae_model, epoches=15, lr=1e-3)
    torch.save(ae_model.to_dict(), save_folder + "ae_4_fix.pth")

