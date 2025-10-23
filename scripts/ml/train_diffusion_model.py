"""
Continuous diffusion model for conditional sparse voxel generation (16x16x16)
Conditioning: CLIP (ViT-B/32) embeddings
Approach: Gaussian diffusion over per-voxel learned texture embeddings + occupancy
Architecture: 3D U-Net with transformer bottleneck and cross-attention to CLIP
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
POSITIONAL_ENCODING_DIM: int = 30


# ----------------------------- Utilities ---------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B,1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:  # pad
            emb = F.pad(emb, (0, 1))
        return emb


def exists(x):
    return x is not None


# Small MLP used often
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_mult=4, act=nn.SiLU()):
        super().__init__()
        h = in_dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            act,
            nn.Linear(h, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------- 3D ResBlock --------------------------------

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, cond_dim=None, groups=8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim

        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)

        if exists(time_emb_dim):
            self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        else:
            self.time_mlp = None

        if exists(cond_dim):
            self.cond_mlp = nn.Linear(cond_dim, out_ch)
        else:
            self.cond_mlp = None

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        if in_ch != out_ch:
            self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb=None, cond=None):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        if exists(t_emb) and self.time_mlp is not None:
            t_add = self.time_mlp(t_emb)[:, :, None, None, None]
            h = h + t_add

        if exists(cond) and self.cond_mlp is not None:
            c_add = self.cond_mlp(cond)[:, :, None, None, None]
            h = h + c_add

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.skip(x)


# ----------------------------- Attention blocks ---------------------------

class CrossAttention(nn.Module):
    """Multi-head attention where keys/values come from condition (CLIP)
    Query: spatial tokens flattened; Key/Value: cond tokens (can be single vector expanded)
    """

    def __init__(self, dim, cond_dim, heads=8, dim_head=64):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, cond):
        # x: (B, N, dim)
        # cond: (B, M, cond_dim)  (M can be 1)
        b, n, _ = x.shape
        m = cond.shape[1]
        h = self.heads

        q = self.to_q(x).view(b, n, h, self.dim_head).transpose(1, 2)  # (B, h, n, dh)
        k = self.to_k(cond).view(b, m, h, self.dim_head).transpose(1, 2)  # (B, h, m, dh)
        v = self.to_v(cond).view(b, m, h, self.dim_head).transpose(1, 2)  # (B, h, m, dh)

        attn = torch.einsum('bhnd,bhmd->bhnm', q, k) * (1.0 / math.sqrt(self.dim_head))
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, h * self.dim_head)
        return self.to_out(out)


class SimpleSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        # x: (B, N, dim)
        b, n, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, h, self.dim_head).transpose(1, 2) for t in qkv]
        attn = torch.einsum('bhnd,bhmd->bhnm', q, k) * (1.0 / math.sqrt(self.dim_head))
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, h * self.dim_head)
        return self.to_out(out)


# ----------------------------- Transformer bottleneck ---------------------

class TransformerBottleneck(nn.Module):
    def __init__(self, dim, depth, cond_dim, heads=8, dim_head=64):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                SimpleSelfAttention(dim, heads=heads, dim_head=dim_head),
                nn.LayerNorm(dim),
                CrossAttention(dim, cond_dim, heads=heads, dim_head=dim_head),
                nn.LayerNorm(dim),
                nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
            ]))

    def forward(self, x, cond):
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


# ----------------------------- UNet3D ------------------------------------

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class UNet3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=128,
                 channel_mults=(1, 2, 4),
                 time_emb_dim=512,
                 cond_dim=512,
                 transformer_depth=4):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # project conditioning (CLIP 512 -> cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(LABEL_EMBEDDING_DIMENSIONS, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        chs = [base_channels * m for m in channel_mults]
        self.inc = nn.Conv3d(in_channels, chs[0], kernel_size=3, padding=1)

        # down path
        self.downs = nn.ModuleList()
        in_ch = chs[0]
        for out_ch in chs:
            self.downs.append(nn.ModuleList([
                ResBlock3D(in_ch, out_ch, time_emb_dim=time_emb_dim, cond_dim=cond_dim),
                ResBlock3D(out_ch, out_ch, time_emb_dim=time_emb_dim, cond_dim=cond_dim),
                Downsample(out_ch, out_ch)
            ]))
            in_ch = out_ch

        # bottleneck
        self.bot1 = ResBlock3D(in_ch, in_ch * 2, time_emb_dim=time_emb_dim, cond_dim=cond_dim)
        self.transformer = TransformerBottleneck(in_ch * 2, depth=transformer_depth, cond_dim=cond_dim)
        self.bot2 = ResBlock3D(in_ch * 2, in_ch, time_emb_dim=time_emb_dim, cond_dim=cond_dim)

        # up path
        self.ups = nn.ModuleList()
        for out_ch, skip_ch in zip(reversed(chs), reversed(chs[:-1] + [chs[-1]])):
            self.ups.append(nn.ModuleList([
                Upsample(in_ch, out_ch),
                ResBlock3D(out_ch + skip_ch, out_ch, time_emb_dim=time_emb_dim, cond_dim=cond_dim),
                ResBlock3D(out_ch, out_ch, time_emb_dim=time_emb_dim, cond_dim=cond_dim)
            ]))
            in_ch = out_ch

        # final conv
        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv3d(in_ch, out_channels, kernel_size=1)

    def forward(self, x, t, clip_emb):
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
        i = 0
        for (up, res1, res2), skip in zip(self.ups, reversed(hs)):
            i += 1
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


# ----------------------------- Positional Encoding -------------------------

def get_3d_sincos_pos_embed(grid_size=16, dim=POSITIONAL_ENCODING_DIM, device=None):
    """
    Generate 3D sinusoidal positional encoding
    Returns: (dim, grid_size, grid_size, grid_size)
    """
    assert dim % 6 == 0, f'Positional encoding dim must be divisible by 6, got {dim}'

    dim_per_axis = dim // 3
    half = dim_per_axis // 2

    emb = torch.zeros(dim, grid_size, grid_size, grid_size, device=device)

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

        s = torch.sin(base)
        c = torch.cos(base)

        emb[i * dim_per_axis: i * dim_per_axis + half] = s
        emb[i * dim_per_axis + half: i * dim_per_axis + 2 * half] = c

    return emb


# ----------------------------- Diffusion process -------------------------

class GaussianDiffusion:
    def __init__(self,
                 model: nn.Module,
                 E_tex: nn.Embedding,
                 num_timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 2e-2,
                 device: Optional[torch.device] = None,
                 unconditional_prob: float = 0.1):
        self.model = model
        self.E_tex = E_tex

        self.K, self.D = E_tex.weight.shape
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
        self.pos_encoding = get_3d_sincos_pos_embed(grid_size=16, dim=POSITIONAL_ENCODING_DIM,
                                                     device=self.device)

    def build_model_input(self, x_emb: torch.Tensor, occ_mask: torch.Tensor) -> torch.Tensor:
        """
        Build model input by concatenating embeddings, occupancy, and positional encoding
        Args:
            x_emb: (B, D, 16, 16, 16) - texture embeddings (possibly noised)
            occ_mask: (B, 1, 16, 16, 16) - occupancy mask
        Returns:
            (B, D + 1 + pos_dim, 16, 16, 16)
        """
        B = x_emb.shape[0]

        # Add positional encoding
        pos = self.pos_encoding.unsqueeze(0).expand(B, -1, -1, -1, -1)

        # Concatenate: [embeddings, occupancy, positional encoding]
        model_input = torch.cat([x_emb, occ_mask, pos], dim=1)

        return model_input

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        # x_start: (B, D, 16,16,16)
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus * noise

    def training_loss(self, batch: dict) -> torch.Tensor:
        # batch must have: 'occupancy' (B,1,16,16,16), 'texture_id' (B,16,16,16), 'clip_embedding' (B,512)
        occ = batch['occupancy'].float().to(self.device)
        tex_id = batch['texture_id'].long().to(self.device)
        clip_emb = batch['clip_embedding'].to(self.device)

        B = occ.shape[0]

        # Build x0 embedding field
        E = self.E_tex.weight  # (K, D)
        x0 = F.embedding(tex_id, E)  # (B,16,16,16,D)
        x0 = x0.permute(0, 4, 1, 2, 3).contiguous()  # (B,D,16,16,16)

        # Set empty voxels to learned empty embedding (index K-1)
        E_empty = E[-1].view(1, self.D, 1, 1, 1)
        x0 = x0 * occ + E_empty * (1.0 - occ)

        # Sample timestep
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)

        # Build model input with positional encoding
        model_input = self.build_model_input(x_t, occ)

        # Apply unconditional dropout
        do_uncond = (torch.rand(B, device=self.device) < self.unconditional_prob)
        clip_in = clip_emb.clone()
        clip_in[do_uncond] = 0.0

        # Model predicts epsilon
        eps_pred = self.model(model_input, t, clip_in)

        # Compute loss (MSE between predicted and actual noise)
        loss = F.mse_loss(eps_pred, noise, reduction='none')
        loss = loss.mean(dim=1, keepdim=True)  # avg over embedding dim

        # Weight loss by occupancy
        loss = (loss * occ).sum() / (occ.sum() + 1e-8)

        return loss

    @torch.no_grad()
    def p_sample_loop(self,
                      cond_clip: torch.Tensor,
                      occ_prior: Optional[torch.Tensor] = None,
                      steps: Optional[int] = None,
                      guidance_scale: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the diffusion model
        Args:
            cond_clip: (B, 512) - CLIP embeddings
            occ_prior: (B, 1, 16, 16, 16) - optional occupancy prior
            steps: number of diffusion steps (default: num_timesteps)
            guidance_scale: classifier-free guidance scale
        Returns:
            texture_ids: (B, 16, 16, 16)
            occupancy: (B, 16, 16, 16)
        """
        device = self.device
        B = cond_clip.shape[0]
        steps = steps or self.num_timesteps

        # Start from pure gaussian noise
        x_t = torch.randn(B, self.D, 16, 16, 16, device=device)

        # Use occupancy prior or assume fully occupied
        if occ_prior is None:
            occ_prior = torch.ones(B, 1, 16, 16, 16, device=device)
        else:
            occ_prior = occ_prior.to(device)

        # Sampling loop
        timesteps = torch.linspace(self.num_timesteps - 1, 0, steps, device=device).long()

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Build model inputs
            model_input_cond = self.build_model_input(x_t, occ_prior)
            model_input_uncond = self.build_model_input(x_t, occ_prior)

            # Conditional prediction
            eps_cond = self.model(model_input_cond, t_batch, cond_clip)

            # Unconditional prediction
            null_clip = torch.zeros_like(cond_clip)
            eps_uncond = self.model(model_input_uncond, t_batch, null_clip)

            # Classifier-free guidance
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # DDPM update
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
            beta_t = self.betas[t]

            # Predict x0
            x0_pred = (x_t - (1 - alpha_t).sqrt() * eps) / (alpha_t.sqrt() + 1e-8)

            if t > 0:
                noise = torch.randn_like(x_t)
                sigma = ((1 - alpha_prev) / (1 - alpha_t + 1e-8) * beta_t).sqrt()
                x_t = alpha_prev.sqrt() * x0_pred + sigma * noise
            else:
                x_t = x0_pred

        # Map embeddings to nearest texture IDs
        E = self.E_tex.weight  # (K, D)

        # Reshape for distance computation
        vox = x_t.permute(0, 2, 3, 4, 1).contiguous()  # (B,16,16,16,D)
        vox_flat = vox.view(B, -1, self.D)  # (B, N, D)

        # Compute distances to all embeddings
        dists = torch.cdist(vox_flat, E)  # (B, N, K)

        # Find nearest embedding
        ids = dists.argmin(dim=-1)  # (B, N)
        ids = ids.view(B, 16, 16, 16)

        # Determine occupancy: compare distance to empty embedding vs others
        empty_idx = self.K - 1
        empty_dists = dists[..., empty_idx]  # (B, N)
        min_dists = dists.min(dim=-1)[0]  # (B, N)

        # Voxel is occupied if its closest embedding is NOT the empty one
        occ_mask = (ids != empty_idx).float()
        occ_mask = occ_mask.view(B, 16, 16, 16)

        return ids, occ_mask


# ----------------------------- Model wrapper -----------------------------

class VoxelDiffusionModel(nn.Module):
    def __init__(
            self,
            embedding_dim: int = TEXTURE_EMBEDDING_DIMENSIONS,
            base_channels: int = 64,
            channel_mults=(1, 2, 4),
            time_emb_dim: int = 512,
            cond_dim: int = 512,
            transformer_depth: int = 4
    ):
        super().__init__()

        # Input: embedding (D) + occupancy (1) + positional encoding (pos_dim)
        in_channels = embedding_dim + 1 + POSITIONAL_ENCODING_DIM
        out_channels = embedding_dim

        self.unet = UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            time_emb_dim=time_emb_dim,
            cond_dim=cond_dim,
            transformer_depth=transformer_depth
        )

    def forward(self, x, t, clip_emb):
        # x: (B, C_in, 16,16,16) where C_in = D + 1 + pos_dim
        return self.unet(x, t, clip_emb)


# ----------------------------- Dataset -----------------------------

class VoxelDataset(Dataset):
    def __init__(self, voxel_grids: list, clip_embeddings: torch.Tensor):
        self.data = torch.tensor(np.array(voxel_grids)).long()  # [N, 16, 16, 16]
        self.clip_embeddings = clip_embeddings  # [N, LABEL_EMBED_DIMENSIONS]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        voxel_grid = self.data[idx]  # [16, 16, 16]
        occupancy_grid = (voxel_grid > 0).float().unsqueeze(0)  # [1, 16, 16, 16]
        texture_grid = voxel_grid.long()

        return {
            'occupancy': occupancy_grid,
            'texture_id': texture_grid,
            'clip_embedding': self.clip_embeddings[idx]
        }


# ----------------------------- Training -----------------------------

def train_diffusion_model(num_epochs=10, batch_size=8, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize texture embedding table
    texture_embedding_table = nn.Embedding(
        NUM_TEXTURES,
        TEXTURE_EMBEDDING_DIMENSIONS,
        max_norm=1.0
    ).to(device)

    # Initialize model and diffusion
    model = VoxelDiffusionModel(
        embedding_dim=TEXTURE_EMBEDDING_DIMENSIONS,
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_emb_dim=512,
        cond_dim=512,
        transformer_depth=4
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        texture_embedding_table,
        num_timesteps=1000,
        device=device,
        unconditional_prob=0.1
    )

    # Initialize optimizer
    optimizer = optim.AdamW(
        list(model.parameters()) + list(texture_embedding_table.parameters()),
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

    # Load dataset
    print("Loading training dataset...")
    text_labels, voxel_data_np = get_random_training_dataset(512)

    # Convert labels to CLIP embeddings
    print("Computing CLIP embeddings...")
    with torch.no_grad():
        clip_embeddings = torch.stack([get_clip_embedding(lbl) for lbl in text_labels])

    dataset = VoxelDataset(voxel_data_np, clip_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Starting training for {num_epochs} epochs on {len(dataset)} samples...")
    print(f"Batch size: {batch_size}, Learning rate: {lr}")
    print(f"Texture embedding dim: {TEXTURE_EMBEDDING_DIMENSIONS}")
    print(f"Positional encoding dim: {POSITIONAL_ENCODING_DIM}")
    print(f"Number of textures: {NUM_TEXTURES}")

    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Compute loss
            loss = diffusion.training_loss(batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(texture_embedding_table.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'texture_embedding': texture_embedding_table.state_dict(),
            'optimizer': optimizer.state_dict(),
            'clip_cache': clip_cache,
            'loss': avg_loss,
            'config': {
                'num_textures': NUM_TEXTURES,
                'texture_embedding_dim': TEXTURE_EMBEDDING_DIMENSIONS,
                'positional_encoding_dim': POSITIONAL_ENCODING_DIM,
                'label_embedding_dim': LABEL_EMBEDDING_DIMENSIONS
            }
        }, f"data/model/diffusion-checkpoint-epoch{epoch + 1}.pth")

    print("Training complete!")
    return model, diffusion, texture_embedding_table


# ----------------------------- Inference -----------------------------

@torch.no_grad()
def generate_voxels(prompt: str,
                   model: VoxelDiffusionModel,
                   diffusion: GaussianDiffusion,
                   clip_model,
                   device: torch.device,
                   guidance_scale: float = 2.0,
                   num_samples: int = 1,
                   steps: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate voxel grids from text prompt

    Args:
        prompt: Text description
        model: Trained diffusion model
        diffusion: GaussianDiffusion instance
        clip_model: CLIP model for encoding text
        device: torch device
        guidance_scale: Classifier-free guidance scale
        num_samples: Number of samples to generate
        steps: Number of diffusion steps

    Returns:
        texture_ids: (num_samples, 16, 16, 16) - texture IDs
        occupancy: (num_samples, 16, 16, 16) - occupancy mask
    """
    model.eval()

    # Encode prompt with CLIP
    tokens = clip.tokenize([prompt]).to(device)
    clip_emb = clip_model.encode_text(tokens).float()  # (1, 512)

    # Expand for multiple samples
    clip_emb = clip_emb.repeat(num_samples, 1)  # (num_samples, 512)

    # Generate
    texture_ids, occupancy = diffusion.p_sample_loop(
        clip_emb,
        occ_prior=None,
        steps=steps,
        guidance_scale=guidance_scale
    )

    return texture_ids, occupancy


# if __name__ == "__main__":
#     # Train the model
#     model, diffusion, texture_embedding_table = train_diffusion_model(
#         num_epochs=10,
#         batch_size=8,
#         lr=1e-4
#     )
#
#     # Example: Generate voxels from a prompt
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     clip_model, _ = clip.load("ViT-B/32", device=device)
#     clip_model.eval()
#
#     prompt = "a red house with a wooden door"
#     texture_ids, occupancy = generate_voxels(
#         prompt=prompt,
#         model=model,
#         diffusion=diffusion,
#         clip_model=clip_model,
#         device=device,
#         guidance_scale=2.0,
#         num_samples=1,
#         steps=50
#     )
#
#     print(f"Generated voxel grid with shape: {texture_ids.shape}")
#     print(f"Occupancy: {occupancy.sum().item()} / {occupancy.numel()} voxels")