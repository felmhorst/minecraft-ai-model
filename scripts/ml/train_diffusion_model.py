"""
Continuous diffusion model for conditional sparse voxel generation (16x16x16)
Conditioning: CLIP (ViT-B/32) embeddings
Approach: Gaussian diffusion over per-voxel learned texture embeddings + occupancy
Architecture: 3D U-Net with transformer bottleneck and cross-attention to CLIP

IMPROVEMENTS:
- Separated occupancy and texture losses with independent heads
- Reduced model capacity (base_channels: 128->64, transformer depth: 4->2)
- Binary cross-entropy for occupancy
- Heavily weighted occupancy loss (50x)
- Added dropout for regularization
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import clip
from torch.utils.data import DataLoader, Dataset
from scripts.training_data.get_random_training_dataset import get_random_training_dataset
from scripts.training_data.normalize_block_ids import get_max_block_id

NUM_TEXTURES: int = get_max_block_id() + 1
TEXTURE_EMBEDDING_DIMENSIONS: int = 8
LABEL_EMBEDDING_DIMENSIONS: int = 512  # CLIP ViT-B/32 output size
POSITIONAL_ENCODING_DIMENSIONS: int = 30


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(
            self,
            dimensions: int):
        super().__init__()
        self.dimensions = dimensions

    def forward(
            self,
            t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B,1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half = self.dimensions // 2
        embedding = math.log(10000) / (half - 1)
        embedding = torch.exp(torch.arange(half, device=t.device) * -embedding)
        embedding = t.float() * embedding[None, :]
        embedding = torch.cat([embedding.sin(), embedding.cos()], dim=-1)
        if self.dimensions % 2 == 1:  # pad
            embedding = F.pad(embedding, (0, 1))
        return embedding


def exists(x):
    return x is not None


# Small MLP used often
class MLP(nn.Module):
    def __init__(
            self,
            in_dimensions: int,
            out_dimensions: int,
            hidden_multiplier: int = 4,
            activation=nn.SiLU()):
        super().__init__()
        hidden_dimensions = in_dimensions * hidden_multiplier
        self.net = nn.Sequential(
            nn.Linear(in_dimensions, hidden_dimensions),
            activation,
            nn.Linear(hidden_dimensions, out_dimensions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            time_embedding_dimensions=None,
            conditional_dimensions=None,
            groups=8,
            dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_dimensions = time_embedding_dimensions
        self.conditional_dimensions = conditional_dimensions

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        if exists(time_embedding_dimensions):
            self.time_mlp = nn.Linear(time_embedding_dimensions, out_channels)
        else:
            self.time_mlp = None

        if exists(conditional_dimensions):
            self.cond_mlp = nn.Linear(conditional_dimensions, out_channels)
        else:
            self.cond_mlp = None

        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            t_emb=None,
            cond=None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        if exists(t_emb) and self.time_mlp is not None:
            t_add = self.time_mlp(t_emb)[:, :, None, None, None]
            h = h + t_add

        if exists(cond) and self.cond_mlp is not None:
            c_add = self.cond_mlp(cond)[:, :, None, None, None]
            h = h + c_add

        h = self.dropout(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.skip(x)


class CrossAttention(nn.Module):
    """Multi-head attention where keys/values come from condition (CLIP)
    Query: spatial tokens flattened; Key/Value: cond tokens (can be single vector expanded)
    """

    def __init__(
            self,
            dim,
            conditional_dimensions,
            heads=8,
            dim_head=64,
            dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(conditional_dimensions, inner_dim, bias=False)
        self.to_v = nn.Linear(conditional_dimensions, inner_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor, cond) -> torch.Tensor:
        # x: (B, N, dim)
        # cond: (B, M, conditional_dimensions)  (M can be 1)
        b, n, _ = x.shape
        m = cond.shape[1]
        h = self.heads

        q = self.to_q(x).view(b, n, h, self.dim_head).transpose(1, 2)  # (B, h, n, dh)
        k = self.to_k(cond).view(b, m, h, self.dim_head).transpose(1, 2)  # (B, h, m, dh)
        v = self.to_v(cond).view(b, m, h, self.dim_head).transpose(1, 2)  # (B, h, m, dh)

        attn = torch.einsum('bhnd,bhmd->bhnm', q, k) * (1.0 / math.sqrt(self.dim_head))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, h * self.dim_head)
        return self.to_out(out)


class SimpleSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, dim)
        b, n, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, h, self.dim_head).transpose(1, 2) for t in qkv]
        attn = torch.einsum('bhnd,bhmd->bhnm', q, k) * (1.0 / math.sqrt(self.dim_head))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, h * self.dim_head)
        return self.to_out(out)


class TransformerBottleneck(nn.Module):
    def __init__(self, dim, depth, conditional_dimensions, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                SimpleSelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),
                CrossAttention(dim, conditional_dimensions, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),
                nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
            ]))

    def forward(self, x: torch.Tensor, cond) -> torch.Tensor:
        # x: (B, C, r, r, r) -> flatten to tokens
        b, c, r1, r2, r3 = x.shape
        n = r1 * r2 * r3
        x = x.view(b, c, n).permute(0, 2, 1).contiguous()  # (B, N, C)

        for ln1, attn, ln2, cross, ln3, mlp in self.layers:
            x = ln1(x)
            x = attn(x) + x
            x = ln2(x)
            x = cross(x, cond) + x
            x = ln3(x)
            x = mlp(x) + x

        x = x.permute(0, 2, 1).view(b, c, r1, r2, r3)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.op = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.op = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class UNet3D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 base_channels: int = 64,
                 channel_mults: tuple[int] = (1, 2, 4),
                 time_embedding_dimensions: int = 512,
                 conditional_dimensions: int = 512,
                 transformer_depth: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.time_embedding_dimensions = time_embedding_dimensions
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_embedding_dimensions),
            nn.Linear(time_embedding_dimensions, time_embedding_dimensions),
            nn.SiLU(),
            nn.Linear(time_embedding_dimensions, time_embedding_dimensions)
        )

        # project conditioning (CLIP 512 -> conditional_dimensions)
        self.cond_proj = nn.Sequential(
            nn.Linear(LABEL_EMBEDDING_DIMENSIONS, conditional_dimensions),
            nn.SiLU(),
            nn.Linear(conditional_dimensions, conditional_dimensions)
        )

        channels = [base_channels * m for m in channel_mults]
        self.inc = nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1)

        # down path
        self.downs = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels:
            self.downs.append(nn.ModuleList([
                ResBlock3D(in_ch, out_ch, time_embedding_dimensions=time_embedding_dimensions,
                           conditional_dimensions=conditional_dimensions, dropout=dropout),
                ResBlock3D(out_ch, out_ch, time_embedding_dimensions=time_embedding_dimensions,
                           conditional_dimensions=conditional_dimensions, dropout=dropout),
                Downsample(out_ch, out_ch)
            ]))
            in_ch = out_ch

        # bottleneck
        self.bot1 = ResBlock3D(in_ch, in_ch * 2, time_embedding_dimensions=time_embedding_dimensions,
                               conditional_dimensions=conditional_dimensions, dropout=dropout)
        self.transformer = TransformerBottleneck(in_ch * 2, depth=transformer_depth,
                                                 conditional_dimensions=conditional_dimensions, dropout=dropout)
        self.bot2 = ResBlock3D(in_ch * 2, in_ch, time_embedding_dimensions=time_embedding_dimensions,
                               conditional_dimensions=conditional_dimensions, dropout=dropout)

        # up path
        self.ups = nn.ModuleList()
        for out_ch, skip_ch in zip(reversed(channels), reversed(channels[:-1] + [channels[-1]])):
            self.ups.append(nn.ModuleList([
                Upsample(in_ch, out_ch),
                ResBlock3D(out_ch + skip_ch, out_ch, time_embedding_dimensions=time_embedding_dimensions,
                          conditional_dimensions=conditional_dimensions, dropout=dropout),
                ResBlock3D(out_ch, out_ch, time_embedding_dimensions=time_embedding_dimensions,
                          conditional_dimensions=conditional_dimensions, dropout=dropout)
            ]))
            in_ch = out_ch

        # final conv
        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv3d(in_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t, clip_emb) -> torch.Tensor:
        # x: (B, C_in, 16,16,16)
        t_emb = self.time_mlp(t)
        cond = self.cond_proj(clip_emb)
        # expand cond to tokens
        cond_tokens = cond.unsqueeze(1)

        hs = []
        h = self.inc(x)
        # down
        for res1, res2, down in self.downs:
            h = res1(h, t_emb=t_emb, cond=cond)
            h = res2(h, t_emb=t_emb, cond=cond)
            hs.append(h)
            h = down(h)

        # bottleneck
        h = self.bot1(h, t_emb=t_emb, cond=cond)
        h = self.transformer(h, cond_tokens)
        h = self.bot2(h, t_emb=t_emb, cond=cond)

        # up
        for (up, res1, res2), skip in zip(self.ups, reversed(hs)):
            h = up(h)
            # align in case of shape mismatch
            if h.shape[-3:] != skip.shape[-3:]:
                h = center_crop_3d(h, skip.shape[-3:])
            h = torch.cat([h, skip], dim=1)
            h = res1(h, t_emb=t_emb, cond=cond)
            h = res2(h, t_emb=t_emb, cond=cond)

        h = self.out_norm(h)
        h = self.out_act(h)
        out = self.out_conv(h)
        return out


def center_crop_3d(x, target_shape):
    _, _, d1, d2, d3 = x.shape
    td1, td2, td3 = target_shape
    sd1 = (d1 - td1) // 2
    sd2 = (d2 - td2) // 2
    sd3 = (d3 - td3) // 2
    return x[:, :, sd1:sd1 + td1, sd2:sd2 + td2, sd3:sd3 + td3]


def get_3d_sincos_pos_embed(
        grid_size: int = 16,
        dimensions: int = POSITIONAL_ENCODING_DIMENSIONS,
        device=None) -> torch.Tensor:
    """
    Generate 3D sinusoidal positional encoding
    Returns: (dim, grid_size, grid_size, grid_size)
    """
    assert dimensions % 6 == 0, f'Positional encoding dim must be divisible by 6, got {dimensions}'

    dim_per_axis = dimensions // 3
    half = dim_per_axis // 2

    emb = torch.zeros(dimensions, grid_size, grid_size, grid_size, device=device)

    # Create frequency bands
    freq = torch.arange(half, device=device).float()
    freq = 1.0 / (10000 ** (freq / float(half)))

    # Create coordinate grids normalized to [-1, 1]
    coords = torch.linspace(-1, 1, grid_size, device=device)
    xs, ys, zs = torch.meshgrid(coords, coords, coords, indexing='ij')

    # Apply sinusoidal encoding for each axis
    for i, grid in enumerate([xs, ys, zs]):
        # grid: (grid_size, grid_size, grid_size)
        # freq: (half,)
        base = torch.einsum('i,jkl->ijkl', freq, grid)  # (half, grid_size, grid_size, grid_size)

        sin = torch.sin(base)
        cos = torch.cos(base)

        emb[i * dim_per_axis: i * dim_per_axis + half] = sin
        emb[i * dim_per_axis + half: i * dim_per_axis + 2 * half] = cos

    return emb


class GaussianDiffusion:
    def __init__(self,
                 model: nn.Module,
                 num_timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 2e-2,
                 device: Optional[torch.device] = None,
                 unconditional_prob: float = 0.1):
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # precompute common terms
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]], dim=0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.unconditional_prob = unconditional_prob

        # Cache positional encoding
        self.pos_encoding = get_3d_sincos_pos_embed(grid_size=16, dimensions=POSITIONAL_ENCODING_DIMENSIONS,
                                                    device=self.device)

    def build_model_input(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Build model input by concatenating noised occupancy and positional encoding
        Args:
            x_t: (B, 1, 16, 16, 16) - noised occupancy (continuous values)
        Returns:
            (B, 1 + pos_dim, 16, 16, 16)
        """
        batch_size = x_t.shape[0]

        # Add positional encoding
        pos = self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

        # Concatenate: [noised occupancy, positional encoding]
        model_input = torch.cat([x_t, pos], dim=1)

        return model_input

    def q_sample(
            self,
            x_start: torch.Tensor,
            t: torch.Tensor,
            noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: add noise to occupancy
        Args:
            x_start: (B, 1, 16, 16, 16) - binary occupancy {0, 1} (normalized to [-1, 1])
            t: (B,) - timestep
            noise: (B, 1, 16, 16, 16) - gaussian noise
        Returns:
            x_t: (B, 1, 16, 16, 16) - noised occupancy
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Normalize occupancy from [0,1] to [-1,1] for better diffusion dynamics
        x_normalized = x_start * 2.0 - 1.0

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_normalized + sqrt_one_minus * noise

    def training_loss(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        """
        Compute diffusion training loss for occupancy prediction
        Args:
            batch: dict with 'occupancy' (B,1,16,16,16) and 'clip_embedding' (B,512)
        Returns:
            loss: scalar tensor
            loss_dict: dict with loss components
        """
        occ = batch['occupancy'].float().to(self.device)  # (B, 1, 16, 16, 16)
        clip_emb = batch['clip_embedding'].to(self.device)

        batch_size = occ.shape[0]

        # Sample timestep
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()

        # Add noise to occupancy
        noise = torch.randn_like(occ)
        x_t = self.q_sample(occ, t, noise=noise)

        # Build model input with positional encoding
        model_input = self.build_model_input(x_t)

        # Apply unconditional dropout for classifier-free guidance
        do_uncond = (torch.rand(batch_size, device=self.device) < self.unconditional_prob)
        clip_in = clip_emb.clone()
        clip_in[do_uncond] = 0.0

        # Model predicts noise
        eps_pred = self.model(model_input, t, clip_in)  # (B, 1, 16, 16, 16)

        # MSE loss on noise prediction
        loss = F.mse_loss(eps_pred, noise)

        loss_dict = {
            'total': loss.item(),
            'occupancy': loss.item()
        }

        return loss, loss_dict

    @torch.no_grad()
    def p_sample_loop(self,
                      cond_clip: torch.Tensor,
                      steps: Optional[int] = None,
                      guidance_scale: float = 2.0) -> torch.Tensor:
        """
        Sample occupancy grid from the diffusion model
        Args:
            cond_clip: (B, 512) - CLIP embeddings
            steps: number of diffusion steps (default: num_timesteps)
            guidance_scale: classifier-free guidance scale
        Returns:
            occupancy: (B, 16, 16, 16) - binary occupancy grid
        """
        device = self.device
        B = cond_clip.shape[0]
        steps = steps or self.num_timesteps

        # Start from pure gaussian noise
        x_t = torch.randn(B, 1, 16, 16, 16, device=device)

        # Sampling loop
        timesteps = torch.linspace(self.num_timesteps - 1, 0, steps, device=device).long()

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Build model input
            model_input = self.build_model_input(x_t)

            # Conditional prediction
            eps_cond = self.model(model_input, t_batch, cond_clip)

            # Unconditional prediction
            null_clip = torch.zeros_like(cond_clip)
            eps_uncond = self.model(model_input, t_batch, null_clip)

            # Classifier-free guidance
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # DDPM update
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
            beta_t = self.betas[t]

            # Predict x0 (clean occupancy)
            x0_pred = (x_t - (1 - alpha_t).sqrt() * eps) / (alpha_t.sqrt() + 1e-8)

            if t > 0:
                noise = torch.randn_like(x_t)
                sigma = ((1 - alpha_prev) / (1 - alpha_t + 1e-8) * beta_t).sqrt()
                x_t = alpha_prev.sqrt() * x0_pred + sigma * noise
            else:
                x_t = x0_pred

        # Threshold to get binary occupancy (x0_pred is in [-1,1] range)
        # Convert back from [-1,1] to [0,1] then threshold at 0.5
        occupancy = ((x_t + 1.0) / 2.0 > 0.5).float().squeeze(1)  # (B, 16, 16, 16)

        return occupancy


class VoxelDiffusionModel(nn.Module):
    def __init__(
            self,
            base_channels: int = 64,
            channel_mults=(1, 2, 4),
            time_embedding_dimensions: int = 512,
            conditional_dimensions: int = 512,
            transformer_depth: int = 2,
            dropout: float = 0.1
    ):
        super().__init__()

        # Input: occupancy (1) + positional encoding (pos_dim)
        in_channels = 1 + POSITIONAL_ENCODING_DIMENSIONS
        # Output: noise prediction for occupancy (1)
        out_channels = 1

        self.unet = UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            time_embedding_dimensions=time_embedding_dimensions,
            conditional_dimensions=conditional_dimensions,
            transformer_depth=transformer_depth,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, t, clip_emb) -> torch.Tensor:
        # x: (B, C_in, 16,16,16) where C_in = 1 + pos_dim
        return self.unet(x, t, clip_emb)


class VoxelDataset(Dataset):
    def __init__(self, voxel_grids: list[torch.Tensor], clip_embeddings: torch.Tensor):
        """
        Args:
            voxel_grids: list of binary occupancy grids (0/1)
            clip_embeddings: (N, 512) CLIP embeddings
        """
        self.data = torch.tensor(np.array(voxel_grids)).float()  # [N, 16, 16, 16]
        self.clip_embeddings = clip_embeddings  # [N, LABEL_EMBED_DIMENSIONS]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        voxel_grid = self.data[index]  # [16, 16, 16]
        occupancy_grid = voxel_grid.unsqueeze(0)  # [1, 16, 16, 16]

        return {
            'occupancy': occupancy_grid,
            'clip_embedding': self.clip_embeddings[index]
        }


def train_diffusion_model(
        num_epochs: int = 100,
        batch_size: int = 8,
        lr: float = 1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model and diffusion
    model = VoxelDiffusionModel(
        base_channels=64,
        channel_mults=(1, 2, 3),  # Changed from (1,2,4) - less aggressive downsampling
        time_embedding_dimensions=512,
        conditional_dimensions=512,
        transformer_depth=2,
        dropout=0.1
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        num_timesteps=1000,
        device=device,
        unconditional_prob=0.1
    )

    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    # Initialize CLIP
    print("Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    clip_cache = {}

    def get_clip_embedding(text_label: str) -> torch.Tensor:
        if text_label not in clip_cache:
            tokens = clip.tokenize([text_label]).to(device)
            with torch.no_grad():
                clip_cache[text_label] = clip_model.encode_text(tokens).squeeze(0).float()
        return clip_cache[text_label]

    print(f"Batch size: {batch_size}, Learning rate: {lr}")
    print(f"Positional encoding dim: {POSITIONAL_ENCODING_DIMENSIONS}")
    print(f"Base channels: 64, Transformer depth: 2")
    print(f"Dropout: 0.1")
    print("Training occupancy-only diffusion model...")

    for epoch in range(num_epochs):
        text_labels, voxel_data_np = get_random_training_dataset(512)

        # Convert labels to CLIP embeddings
        with torch.no_grad():
            clip_embeddings = torch.stack([get_clip_embedding(lbl) for lbl in text_labels])

        dataset = VoxelDataset(voxel_data_np, clip_embeddings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_loss = 0.0
        model.train()

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Compute loss
            loss, loss_dict = diffusion.training_loss(batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss_dict['total']

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")

        if (epoch + 1) % 100 == 0:
            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'clip_cache': clip_cache,
                'loss': avg_loss,
                'config': {
                    'positional_encoding_dim': POSITIONAL_ENCODING_DIMENSIONS,
                    'label_embedding_dim': LABEL_EMBEDDING_DIMENSIONS,
                    'base_channels': 64,
                    'channel_mults': (1, 2, 3),
                    'transformer_depth': 2,
                    'dropout': 0.1
                }
            }, f"data/model/diffusion-checkpoint-{epoch + 1}.pth")
            print("Checkpoint saved!")

    print("Training complete!")
    return model, diffusion


def load_model(
        file_path: str = "data/model/diffusion-checkpoint-100.pth",
        lr: float = 1e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load checkpoint
    checkpoint = torch.load(file_path)
    epoch = checkpoint['epoch']
    clip_cache = checkpoint['clip_cache']
    config = checkpoint.get('config', {})

    print(f'Loading diffusion model (epoch {epoch})')

    # load model
    model = VoxelDiffusionModel(
        base_channels=config.get('base_channels', 64),
        channel_mults=config.get('channel_mults', (1, 2, 3)),
        time_embedding_dimensions=512,
        conditional_dimensions=512,
        transformer_depth=config.get('transformer_depth', 2),
        dropout=config.get('dropout', 0.1)
    ).to(device)
    model.load_state_dict(checkpoint['model'])

    # load optimiser
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )
    optimizer.load_state_dict(checkpoint["optimizer"])

    # diffusion
    diffusion = GaussianDiffusion(
        model,
        num_timesteps=1000,
        device=device,
        unconditional_prob=0.1
    )

    clip_model, _ = clip.load("ViT-B/32", device=device)

    return model, diffusion, optimizer, clip_model, epoch


# ----------------------------- Inference -----------------------------

@torch.no_grad()
def sample_diffusion_model(prompt: str,
                           guidance_scale: float = 2.0,
                           num_samples: int = 1,
                           steps: int = 50) -> np.ndarray:
    """
    Generate voxel occupancy grids from text prompt

    Args:
        prompt: Text description
        guidance_scale: Classifier-free guidance scale
        num_samples: Number of samples to generate
        steps: Number of diffusion steps

    Returns:
        occupancy: (num_samples, 16, 16, 16) - binary occupancy grid
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, diffusion, optimizer, clip_model, epoch = load_model()

    model.eval()
    clip_model.eval()

    # Encode prompt with CLIP
    tokens = clip.tokenize([prompt]).to(device)
    clip_emb = clip_model.encode_text(tokens).float()  # (1, 512)

    # Expand for multiple samples
    clip_emb = clip_emb.repeat(num_samples, 1)  # (num_samples, 512)

    # Generate
    occupancy = diffusion.p_sample_loop(
        clip_emb,
        steps=steps,
        guidance_scale=guidance_scale
    )

    occupancy = occupancy.cpu().numpy().astype(int)

    return occupancy
    checkpoint = torch.load(file_path)
    epoch = checkpoint['epoch']
    clip_cache = checkpoint['clip_cache']
    config = checkpoint.get('config', {})

    print(f'Loading diffusion model (epoch {epoch})')

    # load model
    model = VoxelDiffusionModel(
        base_channels=config.get('base_channels', 64),
        channel_mults=(1, 2, 4),
        time_embedding_dimensions=512,
        conditional_dimensions=512,
        transformer_depth=config.get('transformer_depth', 2),
        dropout=config.get('dropout', 0.1)
    ).to(device)
    model.load_state_dict(checkpoint['model'])

    # load optimiser
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )
    optimizer.load_state_dict(checkpoint["optimizer"])

    # diffusion
    diffusion = GaussianDiffusion(
        model,
        num_timesteps=1000,
        device=device,
        unconditional_prob=0.1
    )

    clip_model, _ = clip.load("ViT-B/32", device=device)

    return model, diffusion, optimizer, clip_model, epoch


# ----------------------------- Inference -----------------------------

@torch.no_grad()
def sample_diffusion_model(prompt: str,
                           guidance_scale: float = 2.0,
                           num_samples: int = 1,
                           steps: int = 50) -> np.ndarray:
    """
    Generate voxel occupancy grids from text prompt

    Args:
        prompt: Text description
        guidance_scale: Classifier-free guidance scale
        num_samples: Number of samples to generate
        steps: Number of diffusion steps

    Returns:
        occupancy: (num_samples, 16, 16, 16) - binary occupancy grid
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, diffusion, optimizer, clip_model, epoch = load_model()

    model.eval()
    clip_model.eval()

    # Encode prompt with CLIP
    tokens = clip.tokenize([prompt]).to(device)
    clip_emb = clip_model.encode_text(tokens).float()  # (1, 512)

    # Expand for multiple samples
    clip_emb = clip_emb.repeat(num_samples, 1)  # (num_samples, 512)

    # Generate
    occupancy = diffusion.p_sample_loop(
        clip_emb,
        steps=steps,
        guidance_scale=guidance_scale
    )

    occupancy = occupancy.squeeze(0).cpu().numpy().astype(int)

    return occupancy