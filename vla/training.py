import torch
import torch.nn as nn
from tqdm import tqdm

def train_ae_straight(ae, dataloader, iterations=30, lr=1e-4, show_frame_fn=None):
    '''
    Train a block or stack of autoencoders on input reconstruction loss.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ae = ae.to(device)

    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(iterations):
        total_loss = 0.0
        for i, frame in enumerate(tqdm(dataloader)):
            x = frame.to(device)
            xr = ae(x)
            loss = loss_fn(x, xr)
            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()
            total_loss += loss.item()
            if show_frame_fn is not None and i == len(dataloader) - 1:# and epoch % 10 == 0:
                show_frame_fn(frame[0], xr[0])
        print(f"[Epoch {epoch + 1}/{iterations}] Loss: {total_loss/len(dataloader):.6f}")
    return ae.to('cpu')


def train_aes_top(aes, dataloader, decoder_only=True, epochs=20, zsteps=25, lr=1e-4, show_frame_fn=None):
    '''
    Train top layer of stacked autoencoder either with z optimization (decoder only)
    or as an AE block using previous block output reconstruction loss
    (with non-top layers being frozen).
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    aes = aes.to(device)

    ae_layer = aes.aes.pop(-1)
    # Training can be faster with preprocessing,
    # but it spoils iterating over dataloader at each epoch
    #frames_pr = [aes.encode_straight(f.to(device))[-1].detach() for f in dataloader]

    if decoder_only:
        decoder = ae_layer.decoder
        optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(ae_layer.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_loss = 1e+10

    for epoch in range(epochs):
        total_loss = 0.0
        z = None
        for i, frame0 in enumerate(tqdm(dataloader)):
            with torch.no_grad():
                frame = aes.encode_straight(frame0.to(device))[-1].detach()
            if decoder_only:
                # We may want to start with random noise rather than image encoded by random encoder
                # when training decoder only and there is no pretrained encoder
                #if z is None:
                #    z = torch.randn((1, list(decoder.children())[0].in_channels, frame.shape[1]//2, frame.shape[2]//2), requires_grad=True, device=device)
                fake, z, _ = ae_layer.optimize_z(frame, steps=zsteps, inner_lr=3e+3) #, z=z, inner_lr=1e+5 #
                fake = decoder(z.detach())
            else:
                fake = ae_layer(frame)
            if show_frame_fn is not None and i == len(dataloader) - 1 and epoch % 10 == 0:
                show_frame_fn(frame0[0], aes.decode_straight(fake)[-1][0])

            loss = loss_fn(fake, frame)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if total_loss < best_loss:
            best_loss = total_loss
            #torch.save(ae_layer.to('cpu').to_dict(), "ae_layer.pth")
            #ae_layer.to('cuda')

        print(f"[Epoch {epoch + 1}/{epochs}] Loss: {total_loss / len(dataloader):.6f}")

    aes.aes.append(ae_layer)
    return aes.to('cpu')

def train_aes_allz(aes, dataloader, epochs=10, zsteps=20, lr=1e-4, lr_z=3e+3, show_frame_fn=None):
    '''
    Train stacked autoencoders with optimization of z's at all layers.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cnt = len(aes.aes)
    aes = aes.to(device)

    decoder_optimizer = torch.optim.Adam(
        [param for ae in aes.aes for param in ae.get_decoder_params()], lr=lr)
    encoder_optimizers = [torch.optim.Adam(ae.get_encoder_params(), lr=lr) for ae in aes.aes]
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = [0.0]*cnt
        total_eloss = 0.0

        for i, frame in enumerate(tqdm(dataloader)):
            x = frame.to(device)
            zs_opt, _ = aes.optimize_latents(x, steps=zsteps, lr=lr_z)
            xs = [aes.aes[n].decode(zs_opt[n].detach()) for n in range(len(zs_opt))]
            zs_opt = [x]+zs_opt
            losses = [loss_fn(xs[n], zs_opt[n].detach()) for n in range(len(xs))]
            loss = None
            for n, l in enumerate(losses):
                c = xs[n].numel()
                if loss is None:
                    loss = l * c
                    tot_c = c
                else:
                    loss += l * c
                    tot_c += c
            l = l / tot_c
            decoder_optimizer.zero_grad()
            loss.backward()
            decoder_optimizer.step()
            for n in range(len(xs)):
                #zn = ae.aes[n].decode(zs_opt[n+1].detach())
                #enc = ae.aes[n].encode(zn.detach())
                enc = aes.aes[n].encode(zs_opt[n].detach())
                loss_e = loss_fn(enc, zs_opt[n+1].detach())
                encoder_optimizers[n].zero_grad()
                loss_e.backward()
                encoder_optimizers[n].step()
                total_eloss += loss_e.item()
            if show_frame_fn is not None and i == len(dataloader) - 1:# and epoch % 10 == 0:
                show_frame_fn(frame, xs[0])
            for n in range(len(losses)):
                total_loss[n] += losses[n].item()

        s = ""
        for tl in total_loss:
            s += f"{tl / len(dataloader):.6} "
        print(f"[Epoch {epoch + 1}/{epochs}] Loss: ({s}) | {total_eloss/len(dataloader):.6f}")

    return aes.to('cpu')

