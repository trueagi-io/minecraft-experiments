import torch
import torch.nn as nn
import torch.nn.functional as F


class AEModelAbstract(nn.Module):

    def __init__(self):
        super().__init__()

    @classmethod
    def from_dict(cls, mp_dict):
        model = cls(**mp_dict['model_hyperparams'])
        model.load_state_dict(mp_dict['model_state_dict'])
        return model

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def get_encoder_params(self):
        return self.encoder.parameters()

    def get_decoder_params(self):
        return self.decoder.parameters()

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out

    def optimize_z(self, x, steps=25, inner_lr=3e+3, z=None):
        if z is None:
            z = nn.Parameter(self.encode(x).detach())
        latent_optimizer = torch.optim.SGD([z], lr=inner_lr)
        loss_fn = nn.MSELoss()
        for i in range(steps+1):
            x_rec = self.decode(z)
            loss = loss_fn(x_rec, x)
            if i != steps:
                latent_optimizer.zero_grad()
                loss.backward()
                latent_optimizer.step()
        return x_rec, z, loss


class ConvAEBlock(AEModelAbstract):

    def __init__(self, in_channels=3, out_channels=6, kernel_size=6, stride=2, padding=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.Sigmoid() # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.Sigmoid() # nn.ReLU()
        )

    def to_dict(self):
        e = list(self.encoder.children())[0]
        return {
            'model_class': 'ConvAEBlock',
            'model_state_dict': self.state_dict(),
            'model_hyperparams': {
                'in_channels': e.in_channels,
                'out_channels': e.out_channels,
                'kernel_size': e.kernel_size,
                'stride': e.stride,
                'padding': e.padding
            }
        }


class ConvAEBlockSparse(AEModelAbstract):
    def __init__(self,
                 in_channels=3,
                 num_blocks=4,
                 channels_per_block=8,
                 kernel_size=6,
                 stride=2,
                 padding=2):
        super().__init__()

        self.num_blocks = num_blocks
        self.channels_per_block = channels_per_block
        self.total_out_channels = num_blocks * channels_per_block

        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=channels_per_block,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                ),
                #nn.ReLU(),
                #nn.BatchNorm2d(channels_per_block)
            )
            for _ in range(num_blocks)
        ])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.total_out_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.Sigmoid()
        )

    def encode(self, x):
        # x: (B, C_in, H, W) or (C_in, H, W)
        outputs = []
        # dim=0 if B is absent (3D), otherwise dim=1 (4D)
        dim = len(x.shape)-3
        for block in self.encoder_blocks:
            out = block(x)
            out = F.softmax(out, dim=dim)  # softmax per each block on each spatial position
            outputs.append(out)
        encoded = torch.cat(outputs, dim=dim)  # (B, total_out_channels, H', W')
        return encoded

    def get_encoder_params(self):
        return [param for block in self.encoder_blocks for param in block.parameters()]

    def to_dict(self):
        e = list(list(self.encoder_blocks.children())[0].children())[0]
        return {
            'model_class': 'ConvAEBlockSparse',
            'model_state_dict': self.state_dict(),
            'model_hyperparams': {
                'in_channels': e.in_channels,
                'num_blocks': self.num_blocks,
                'channels_per_block': self.channels_per_block,
                #'out_channels': e.out_channels,
                'kernel_size': e.kernel_size,
                'stride': e.stride,
                'padding': e.padding
            }
        }


class Conv3DAEBlock(AEModelAbstract):
    def __init__(self, in_channels=3, out_channels=12, kernel_size=6, stride=2, padding=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.Sigmoid() #nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=out_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.Sigmoid() #nn.ReLU()
        )

    def to_dict(self):
        e = list(self.encoder.children())[0]
        return {
            'model_class': 'Conv3DAEBlock',
            'model_state_dict': self.state_dict(),
            'model_hyperparams': {
                'in_channels': e.in_channels,
                'out_channels': e.out_channels,
                'kernel_size': e.kernel_size,
                'stride': e.stride,
                'padding': e.padding
            }
        }

class FixingAEBlock(AEModelAbstract):
    def __init__(self, ae_main, ae_fix):
        super().__init__()
        self.ae_main = ae_main
        # it is typically redundant to have full fixing AE,
        # but it's more convenient, while we don't provide
        # separate classes for encoders and decoders
        self.ae_fix = ae_fix

    def encode(self, x):
        z0 = self.ae_main.encode(x)
        x1 = self.decode(z0)
        # encoder_fix approximate derivations of decoder over z multiplied by dx
        # these derivations depend on z in general (not on dx)
        # but if the decoder layer is simple (nearly linear), it will weakly depend on z
        # another option would be for encoder_fix to produce a matrix from z or x to be multiplied by dx
        dz = self.ae_fix.encode(x-x1)
        x2 = self.decode(z0+dz)
        return x2

    def decode(self, z):
        return self.ae_main.decode(z)

    def to_dict(self):
        return {
            'model_class': 'FixingAEBlock',
            'ae_main': self.ae_main.to_dict(),
            'ae_fix': self.ae_fix.to_dict()
        }

    @classmethod
    def from_dict(cls, mp_dict):
        ae_main = eval(mp_dict['ae_main']['model_class']).from_dict(mp_dict['ae_main'])
        ae_fix = eval(mp_dict['ae_fix']['model_class']).from_dict(mp_dict['ae_fix'])
        model = cls(ae_main, ae_fix)
        return model


class StackedAE(nn.Module):
    def __init__(self, aes):
        super().__init__()
        self.aes = nn.ModuleList(aes)

    def to_dict(self):
        d = {}
        d['ae_number'] = len(self.aes)
        d['aes'] = {}
        for i in range(d['ae_number']):
            d['aes'][i] = self.aes[i].to_dict()
        return d

    def encode_straight(self, x):
        zs = [x]
        for ae in self.aes:
            zs.append(ae.encode(zs[-1]))
        return zs

    def decode_straight(self, z):
        xs = [z]
        cnt = len(self.aes)
        for i in range(cnt):
            xs.append(self.aes[cnt-i-1].decode(xs[-1]))
        return xs

    def optimize_latents(self, x, zs=None, steps=40, lr=3e+3):
        if zs is None:
            zs = self.encode_straight(x)
        zs = [nn.Parameter(z) for z in zs]
        zs_opt = zs[1:]
        if isinstance(lr, list):
            latent_optimizer = [torch.optim.SGD([z], l) for z, l in zip(zs_opt, lr)]
        else:
            latent_optimizer = torch.optim.SGD(zs_opt, lr)
        cnt = len(self.aes)
        loss_fn = nn.MSELoss()
        for it in range(steps):
            xs = [self.aes[n].decode(zs_opt[n]) for n in range(cnt)]
            tloss = torch.tensor(0.0, device=next(self.parameters()).device)
            for i in range(len(xs)):
                tloss += loss_fn(xs[i], zs[i])
            if isinstance(latent_optimizer, list):
                for lo in latent_optimizer:
                    lo.zero_grad()
                tloss.backward()
                for lo in latent_optimizer:
                    lo.step()
            else:
                latent_optimizer.zero_grad()
                tloss.backward()
                latent_optimizer.step()
            #print(it, tloss.item(), loss_fn(xs[0], zs[0]).item())
        return zs_opt, tloss

    def forward(self, x):
        z = self.encode_straight(x)[-1]
        return self.decode_straight(z)[-1]

    @classmethod
    def from_dict(cls, mp_dict):
        #'model_class'
        aes = [eval(mp_dict['aes'][i].get('model_class', 'ConvAEBlock')).from_dict(mp_dict['aes'][i])
               for i in range(mp_dict['ae_number'])]
        return cls(aes)
