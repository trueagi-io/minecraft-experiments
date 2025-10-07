import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import *

from aeblock import *

def train_aes_straight_T(ae, dataloader, predict=False, epochs=30, lr=1e-4, show_frame_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ae = ae.to(device)

    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for i, frame in enumerate(tqdm(dataloader)):
            x = frame.to(device)
            dx = 1 if predict else 0
            x0 = x[:,:,:-dx,:,:] if predict else x
            z = ae.encode_straight(x0)[-1]
            xr = ae.decode_straight(z)[-1]
            dt = xr.size(2) - z.size(2)
            # By shifting x, we ask AE to reconstruct all x from such z,
            # which received only previous observations for these x
            loss = loss_fn(x[:,:,dt*2+dx:,:,:], xr[:,:,dt:-dt,:,:])
            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()
            total_loss += loss.item()
            if show_frame_fn is not None and i == len(dataloader) - 1:# and epoch % 10 == 0:
                show_frame_fn(frame[0], xr[0])
        print(f"[Epoch {epoch + 1}/{epochs}] Loss: {total_loss/len(dataloader):.6f}")
    return ae.to('cpu')


def train_aes_top_T(ae, dataloader, predict=False, epochs=30, lr=1e-4, show_frame_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ae = ae.to(device)

    ae_optimizer = torch.optim.Adam(ae.aes[-1].parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for i, frame in enumerate(tqdm(dataloader)):
            x = frame.to(device)
            with torch.no_grad():
                for n in range(len(ae.aes)-1):
                    x = ae.aes[n].encode(x)
            dx = 1 if predict else 0
            x0 = x[:,:,:-dx,:,:] if predict else x
            z = ae.aes[-1].encode(x0.detach())
            xr = ae.aes[-1].decode(z)
            dt = xr.size(2)-z.size(2)
            loss = loss_fn(x[:,:,dt*2+dx:,:,:], xr[:,:,dt:-dt,:,:])
            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()
            total_loss += loss.item()
            if show_frame_fn is not None and i == len(dataloader) - 1:# and epoch % 10 == 0:
                show_frame_fn(frame[0], ae.decode_straight(z)[-1][0])
        print(f"[Epoch {epoch + 1}/{epochs}] Loss: {total_loss/len(dataloader):.6f}")
    return ae.to('cpu')


def optimize_latents_inp(aes, x, T_add=16, steps=1000, lr=6e+4):
    '''
    'Inpainting' of missing frames for non-predictive causal 3D AE
    amenable for z optimization. Kinda works, but initialization
    of missing xs (to be predicted) greatly influences initial guesses
    of zs by encoders, and this initial guess is not removed by
    the optimization process 
    '''
    x = x.to('cuda')
    aes = aes.to('cuda')
    T = x.size(1)
    pad = torch.full((x.size(0), T_add, x.size(2), x.size(3)), 0.3, dtype=x.dtype, device=x.device)
    x_padded = torch.cat([x, pad], dim=1)  # (C, T + T_add, H, W)
    zs = aes.encode_straight(x_padded)
    zs = [nn.Parameter(z) for z in zs]
    zs_opt = zs[1:]
    latent_optimizer = torch.optim.SGD(zs_opt, lr)
    cnt = len(aes.aes)
    loss_fn = nn.MSELoss()
    for it in range(steps):
        xs = [aes.aes[n].decode(zs_opt[n]) for n in range(cnt)]
        tloss = torch.tensor(0.0, device=next(aes.parameters()).device)
        for i in range(len(xs)):
            if i == 0:
                tloss += loss_fn(x.detach(), xs[0][:, :T, :, :])
            else:
                tloss += loss_fn(xs[i], zs[i])
        latent_optimizer.zero_grad()
        tloss.backward()
        latent_optimizer.step()
        print(it, tloss.item(), loss_fn(xs[0], zs[0]).item())
        if it == steps-1:
            import time
            #for t in range(T+T_add-1):
            t = 96-16
            show_frame(xs[0].permute(1, 0, 2, 3)[t].detach(), xs[0].permute(1, 0, 2, 3)[t+3].detach())
            time.sleep(5.2)
    return zs_opt, tloss

def show_predict_frame(fake, shift):
    show_frame(fake.permute(1, 0, 2, 3)[-shift-1], fake.permute(1, 0, 2, 3)[-shift])

save_folder = "result/"

if __name__ == "__main__":
    video_path = "data/output.mp4"
    frames = extract_frames(video_path, area=320*240, max_frames=10000)
    dataset = VideoClipDataset(frames, 48+12, 3)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Causal convolution over T without padding. z[0] gets inputs
    # from x[0:ks] meaning that first kernel_size-1 elements are
    # not constructed, because we do not have enough history for them.
    # Then, deconvolution reconstructs all x, but kernel_size-1 elements
    # both from left and right do not get enough inputs from z.
    # This can be interpreted as the reconstruction is x[(kernel_size-1)*2:].
    checkpoint = save_folder + "ae_3tp_3d.pth" # None
    predict = True
    add_layer = False
    multc = 2 if predict else 1
    if checkpoint is None:
        ae_model = StackedAE([
            Conv3DAEBlock(3, 6*multc, kernel_size=(4,6,6), stride=(1,2,2), padding=(0,2,2))
        ])
    else:
        ae_model = StackedAE.from_dict(torch.load(checkpoint, weights_only=True))
        if add_layer:
            e = list(ae_model.aes[-1].encoder.children())[0]
            layer = Conv3DAEBlock(e.out_channels, e.out_channels*2*multc,
                                  kernel_size=(4,6,6), stride=(1,2,2), padding=(0,2,2))
            ae_model.aes.append(layer)
    # train_aes_top_T
    nlost = 3*len(ae_model.aes)
    aes = train_aes_straight_T(ae_model, dataloader, predict=predict, lr=2e-5, epochs=10,
        show_frame_fn=lambda _, fake: show_predict_frame(fake, nlost+1))
    torch.save(aes.to_dict(), save_folder + "ae_3tp_3d.pth")

    if predict:
        import cv2
        for i, clip in enumerate(dataloader):
            out = cv2.VideoWriter(f"result/out{i}.mp4", cv2.VideoWriter_fourcc(*'XVID'), 6, (320, 240))
            with torch.no_grad():
                #clip = clip[:,:,-nlost*2:,:,:]
                for n in range(12):
                    im = clip[0].permute(1, 2, 3, 0)[-12+n].cpu().numpy()
                    out.write(cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    plt.imshow(im)
                    plt.show(block=False)
                    plt.pause(0.2)
                for n in range(9):
                    z = ae_model.encode_straight(clip)[-1]
                    xr = ae_model.decode_straight(z)[-1]
                    dt = xr.size(2)-z.size(2)
                    im = xr[0].permute(1, 2, 3, 0)[-dt-1].cpu().numpy()
                    out.write(cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    plt.imshow(im)
                    plt.show(block=False)
                    plt.pause(0.2)
                    clip = clip[:,:,1:,:,:]
                    clip = torch.cat([clip, xr[:,:,-dt-1:-dt,:,:]], dim=2)
            out.release()
            if i >= 9:
                break
