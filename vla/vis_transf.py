import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class FactorizedVideoTransformer(nn.Module):
    def __init__(self, latent_dim, H, W, num_frames,
                 spatial_depth=2, temporal_depth=2,
                 n_heads=4, pos_dim=16):
        super().__init__()

        self.H = H
        self.W = W
        self.T = num_frames
        self.latent_dim = latent_dim
        self.pos_dim = pos_dim
        self.model_dim = latent_dim + pos_dim  # dimension after spatial embeddings
        self.model_dim2 = latent_dim + 2*pos_dim  # dimension after temporal embeddings

        # Learnable positional embeddings
        self.spatial_pos_emb = nn.Parameter(torch.randn(H * W, pos_dim))
        self.temporal_pos_emb = nn.Parameter(torch.randn(num_frames, pos_dim))

        # Spatial Transformer: input: (HW, latent_dim + pos_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=n_heads, batch_first=True)
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=spatial_depth)

        # Projection into latent_dim after spatial attention -- Not used
        #self.proj_after_spatial = nn.Linear(self.model_dim, latent_dim)

        # Temporal Transformer
        temporal_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim2, nhead=n_heads, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(temporal_encoder_layer, num_layers=temporal_depth)

        # Projection back to latent_dim for output
        self.proj_out = nn.Linear(self.model_dim2, latent_dim)

    def forward(self, x):
        """
        x: Tensor [B, T, H, W, latent_dim]
        return: Tensor [B, T, H, W, latent_dim]
        """
        B, T, H, W, C = x.shape
        assert H == self.H and W == self.W and C == self.latent_dim

        # -------- Spatial Attention --------
        # Flatten spatial tokens: (B, T, H*W, latent_dim)
        x_spatial = x.view(B, T, H * W, C)  # (B, T, HW, C)
        spatial_pos = self.spatial_pos_emb.unsqueeze(0).unsqueeze(0)  # (1,1,HW,pos_dim)
        spatial_pos = spatial_pos.expand(B, T, -1, -1)  # (B, T, HW, pos_dim)
        x_spatial = torch.cat([x_spatial, spatial_pos], dim=-1)  # (B, T, HW, model_dim)

        # Merge batch and time: (B*T, HW, model_dim)
        x_spatial = x_spatial.view(B * T, H * W, self.model_dim)
        x_spatial_out = self.spatial_transformer(x_spatial)  # (B*T, HW, model_dim)

        # Project to latent_dim and reshape back: (B, T, H, W, latent_dim)
        #x_spatial_out = self.proj_after_spatial(x_spatial_out)  # (B*T, HW, latent_dim)
        #x_spatial_out = x_spatial_out.view(B, T, H, W, C)
        x_spatial_out = x_spatial_out.view(B, T, H, W, self.model_dim)

        # -------- Temporal Attention --------
        # permute: (B, T, H, W, C) → (B, H, W, T, C)
        x_temp = x_spatial_out.permute(0, 2, 3, 1, 4).contiguous()  # (B, H, W, T, C)
        x_temp = x_temp.view(B * H * W, T, self.model_dim) # (B*HW, T, C+pos_dim)

        temporal_pos = self.temporal_pos_emb.unsqueeze(0)  # (1, T, pos_dim)
        temporal_pos = temporal_pos.expand(B * H * W, -1, -1)  # (B*HW, T, pos_dim)
        x_temp = torch.cat([x_temp, temporal_pos], dim=-1)  # (B*HW, T, model_dim2)

        x_temp_out = self.temporal_transformer(x_temp)  # (B*HW, T, model_dim2)

        # view back: (B*H*W, T, model_dim2) → (B, T, H, W, model_dim2)
        x_temp_out = x_temp_out.view(B, H, W, T, self.model_dim2).permute(0, 3, 1, 2, 4)  # (B, T, H, W, model_dim)
        # project to latent_dim
        x_out = self.proj_out(x_temp_out)  # (B, T, H, W, latent_dim)

        return x_out

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    from tqdm import tqdm
    #model.train()
    total_loss = 0.0

    for x, target in tqdm(dataloader):
        # x: (B, T, H, W, C),  target: (B, H, W, C)
        x = x.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        out = model(x)  # (B, T, H, W, C)
        # Last frame prediction
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

def train_model(model, train_dataset, val_dataset,
                epochs=10, batch_size=8, lr=1e-4, device='cuda', loss_fn=nn.MSELoss()):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

