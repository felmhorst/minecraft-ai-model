import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import clip


# ============================================================================
# VQ-VAE Components
# ============================================================================
from torch.utils.data import DataLoader, Dataset

from scripts.training_data.get_random_training_dataset import get_random_training_dataset
from scripts.training_data.normalize_block_ids import get_max_block_id


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with EMA updates."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, D, H, W, L) - encoder output
        Returns:
            quantized: (B, D, H, W, L) - quantized embeddings
            indices: (B, H, W, L) - codebook indices
            vq_loss: scalar - VQ loss
        """
        # Flatten spatial dims
        z_flattened = z.permute(0, 2, 3, 4, 1).contiguous()  # (B, H, W, L, D)
        orig_shape = z_flattened.shape
        z_flattened = z_flattened.view(-1, self.embedding_dim)  # (B*H*W*L, D)

        # Calculate distances to codebook entries
        distances = (z_flattened ** 2).sum(dim=1, keepdim=True) + \
                    (self.embedding.weight ** 2).sum(dim=1) - \
                    2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # Get nearest codebook entries
        indices = torch.argmin(distances, dim=1)  # (B*H*W*L,)
        quantized = self.embedding(indices)  # (B*H*W*L, D)

        # Reshape back
        quantized = quantized.view(orig_shape)  # (B, H, W, L, D)
        indices = indices.view(orig_shape[:-1])  # (B, H, W, L)

        # VQ loss
        e_latent_loss = F.mse_loss(quantized.detach(), z_flattened.view(orig_shape))
        q_latent_loss = F.mse_loss(quantized, z_flattened.view(orig_shape).detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = z_flattened.view(orig_shape) + (quantized - z_flattened.view(orig_shape)).detach()
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()  # (B, D, H, W, L)

        return quantized, indices, vq_loss


class VQVAEEncoder(nn.Module):
    """3D Encoder for voxel data."""

    def __init__(self, vocab_size: int, latent_dim: int = 256):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(vocab_size, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU()
        )

        self.conv4 = nn.Conv3d(256, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, vocab_size, 16, 16, 16) - one-hot encoded voxels
        Returns:
            z: (B, latent_dim, 4, 4, 4)
        """
        h = self.conv1(x)  # (B, 64, 8, 8, 8)
        h = self.conv2(h)  # (B, 128, 4, 4, 4)
        h = self.conv3(h)  # (B, 256, 4, 4, 4)
        z = self.conv4(h)  # (B, latent_dim, 4, 4, 4)
        return z


class VQVAEDecoder(nn.Module):
    """3D Decoder for voxel data."""

    def __init__(self, latent_dim: int, vocab_size: int):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(latent_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU()
        )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )

        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )

        self.conv4 = nn.Conv3d(64, vocab_size, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim, 4, 4, 4)
        Returns:
            logits: (B, vocab_size, 16, 16, 16)
        """
        h = self.conv1(z)  # (B, 256, 4, 4, 4)
        h = self.conv2(h)  # (B, 128, 8, 8, 8)
        h = self.conv3(h)  # (B, 64, 16, 16, 16)
        logits = self.conv4(h)  # (B, vocab_size, 16, 16, 16)
        return logits


class VQVAE(nn.Module):
    """Complete VQ-VAE model."""

    def __init__(self, vocab_size: int, num_embeddings: int = 512,
                 embedding_dim: int = 256, commitment_cost: float = 0.25):
        super().__init__()

        self.vocab_size = vocab_size
        self.encoder = VQVAEEncoder(vocab_size, embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = VQVAEDecoder(embedding_dim, vocab_size)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to discrete latent codes."""
        z = self.encoder(x)
        quantized, indices, _ = self.vq(z)
        return quantized, indices

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """Decode from quantized embeddings."""
        return self.decoder(quantized)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from codebook indices."""
        # indices: (B, H, W, L)
        quantized = self.vq.embedding(indices)  # (B, H, W, L, D)
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()  # (B, D, H, W, L)
        return self.decode(quantized)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, vocab_size, 16, 16, 16)
        Returns:
            recon_logits: (B, vocab_size, 16, 16, 16)
            indices: (B, 4, 4, 4)
            vq_loss: scalar
        """
        z = self.encoder(x)
        quantized, indices, vq_loss = self.vq(z)
        recon_logits = self.decoder(quantized)
        return recon_logits, indices, vq_loss


# ============================================================================
# Discrete Absorbing Diffusion Components
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and cross-attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention
        attn_out, _ = self.cross_attn(x, context, context)
        x = self.norm2(x + self.dropout(attn_out))

        # Feedforward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x


class DiffusionTransformer(nn.Module):
    """Transformer-based denoising network for discrete diffusion."""

    def __init__(self, num_embeddings: int, d_model: int = 512, n_layers: int = 6,
                 n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1,
                 clip_dim: int = 512):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.mask_token_id = num_embeddings  # Special mask token

        # Token embedding (includes mask token)
        self.token_embedding = nn.Embedding(num_embeddings + 1, d_model)

        # 3D positional embedding for 4x4x4 grid
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, d_model))

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        # CLIP text projection
        self.text_proj = nn.Linear(clip_dim, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_embeddings + 1)
        )

    def forward(self, indices: torch.Tensor, timesteps: torch.Tensor,
                text_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: (B, H, W, L) - masked token indices (with mask_token_id for masked positions)
            timesteps: (B,) - diffusion timesteps
            text_emb: (B, clip_dim) - CLIP text embeddings
        Returns:
            logits: (B, H*W*L, num_embeddings+1) - predicted logits
        """
        B = indices.shape[0]

        # Flatten spatial dimensions
        indices_flat = indices.view(B, -1)  # (B, 64)

        # Token embeddings
        x = self.token_embedding(indices_flat)  # (B, 64, d_model)

        # Add positional embeddings
        x = x + self.pos_embedding

        # Add timestep embedding
        t_emb = self.time_mlp(timesteps)  # (B, d_model)
        x = x + t_emb.unsqueeze(1)

        # Project text embedding
        text_context = self.text_proj(text_emb).unsqueeze(1)  # (B, 1, d_model)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, text_context)

        # Output logits
        logits = self.output_head(x)  # (B, 64, num_embeddings+1)

        return logits


class DiscreteAbsorbingDiffusion(nn.Module):
    """Discrete absorbing diffusion process."""

    def __init__(self, num_timesteps: int = 1000, schedule: str = 'cosine'):
        super().__init__()

        self.num_timesteps = num_timesteps

        # Create beta schedule (probability of transitioning to mask)
        if schedule == 'cosine':
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        elif schedule == 'linear':
            betas = torch.linspace(0.0001, 0.02, num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1.0 - betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) - forward diffusion process.
        With probability (1 - alpha_bar_t), replace token with mask.

        Args:
            x_0: (B, H, W, L) - original discrete codes
            t: (B,) - timesteps
        Returns:
            x_t: (B, H, W, L) - noised codes
        """
        B, H, W, L = x_0.shape

        # Get alpha_bar for each sample
        alpha_bar = self.alphas_cumprod[t]  # (B,)

        # Sample mask pattern
        mask_prob = 1 - alpha_bar
        mask = torch.rand(B, H, W, L, device=x_0.device) < mask_prob.view(B, 1, 1, 1)

        # Create masked version
        mask_token_id = x_0.max() + 1  # Assume mask token is vocab_size
        x_t = torch.where(mask, torch.full_like(x_0, mask_token_id), x_0)

        return x_t


# ============================================================================
# Complete Model
# ============================================================================

class VoxelDiffusionModel(nn.Module):
    """Complete text-conditional voxel generation model."""

    def __init__(self, vocab_size: int, num_embeddings: int = 512,
                 embedding_dim: int = 256, d_model: int = 512,
                 n_layers: int = 6, n_heads: int = 8, num_timesteps: int = 1000):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_embeddings = num_embeddings
        self.mask_token_id = num_embeddings

        # VQ-VAE
        self.vqvae = VQVAE(vocab_size, num_embeddings, embedding_dim)

        # Diffusion process
        self.diffusion = DiscreteAbsorbingDiffusion(num_timesteps)

        # Denoising network
        self.denoiser = DiffusionTransformer(
            num_embeddings, d_model, n_layers, n_heads,
            d_ff=d_model * 4, clip_dim=512
        )

        # CLIP model (frozen)
        self.clip_model = None  # Will be loaded separately

    def load_clip(self, device: str = 'cuda'):
        """Load CLIP model."""
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_text(self, text: list) -> torch.Tensor:
        """Encode text using CLIP."""
        text_tokens = clip.tokenize(text).to(next(self.parameters()).device)
        text_features = self.clip_model.encode_text(text_tokens).float()
        return text_features

    def compute_vqvae_loss(self, voxels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute VQ-VAE reconstruction loss.

        Args:
            voxels: (B, 16, 16, 16) - voxel texture IDs
        Returns:
            loss, metrics dict
        """
        B = voxels.shape[0]

        # One-hot encode
        x = F.one_hot(voxels.long(), self.vocab_size).float()
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, vocab_size, 16, 16, 16)

        # Forward pass
        recon_logits, indices, vq_loss = self.vqvae(x)

        # Reconstruction loss
        recon_loss = F.cross_entropy(
            recon_logits,  # (B, vocab_size, 16, 16, 16)
            voxels.long()  # (B, 16, 16, 16)
        )

        # Total loss
        total_loss = recon_loss + vq_loss

        metrics = {
            'recon_loss': recon_loss.item(),
            'vq_loss': vq_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, metrics

    def compute_diffusion_loss(self, voxels: torch.Tensor,
                               text_emb: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute diffusion model loss.

        Args:
            voxels: (B, 16, 16, 16) - voxel texture IDs
            text_emb: (B, 512) - CLIP text embeddings
        Returns:
            loss, metrics dict
        """
        B = voxels.shape[0]
        device = voxels.device

        # Encode to discrete codes
        x = F.one_hot(voxels.long(), self.vocab_size).float()
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        _, indices = self.vqvae.encode(x)  # (B, 4, 4, 4)

        # Sample timesteps
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)

        # Apply forward diffusion
        indices_noisy = self.diffusion.q_sample(indices, t)

        # Predict original indices
        logits = self.denoiser(indices_noisy, t.float(), text_emb)

        # Compute loss
        loss = F.cross_entropy(
            logits.view(B, -1, self.num_embeddings + 1).transpose(1, 2),
            indices.view(B, -1).long()
        )

        metrics = {'diffusion_loss': loss.item()}

        return loss, metrics

    @torch.no_grad()
    def sample(self, text: list, num_samples: int = 1,
               device: str = 'cuda') -> torch.Tensor:
        """
        Generate voxels from text.

        Args:
            text: list of text prompts
            num_samples: number of samples per prompt
            device: device to run on
        Returns:
            voxels: (B, 16, 16, 16) - generated voxel texture IDs
        """
        self.eval()

        # Encode text
        text_emb = self.encode_text(text * num_samples)  # (B*num_samples, 512)
        B = text_emb.shape[0]

        # Initialize with all mask tokens
        indices = torch.full((B, 4, 4, 4), self.mask_token_id,
                             dtype=torch.long, device=device)

        # Reverse diffusion
        for t in reversed(range(self.diffusion.num_timesteps)):
            t_batch = torch.full((B,), t, dtype=torch.float, device=device)

            # Predict logits
            logits = self.denoiser(indices, t_batch, text_emb)

            # Sample from categorical distribution
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, self.num_embeddings + 1),
                num_samples=1
            ).view(B, 4, 4, 4)

            # Keep non-mask tokens (absorbing property)
            mask = indices == self.mask_token_id
            indices = torch.where(mask, sampled, indices)

        # Decode to voxels
        voxel_logits = self.vqvae.decode_indices(indices)
        voxels = voxel_logits.argmax(dim=1)  # (B, 16, 16, 16)

        return voxels


# ============================================================================
# Training Functions
# ============================================================================

def train_vqvae(model: VoxelDiffusionModel, optimizer, dataloader, num_epochs: int = 100,
                lr: float = 1e-4, device: str = 'cuda'):
    """Train VQ-VAE."""
    # optimizer = torch.optim.Adam(model.vqvae.parameters(), lr=lr)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch_data in enumerate(dataloader):
            if isinstance(batch_data, (list, tuple)):
                voxels = batch_data[0]
            else:
                voxels = batch_data
            voxels = voxels.to(device)

            optimizer.zero_grad()
            loss, metrics = model.compute_vqvae_loss(voxels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}, Avg Loss: {total_loss / len(dataloader):.3f}")


def train_diffusion(model: VoxelDiffusionModel, optimizer, dataloader, num_epochs: int = 100,
                    lr: float = 1e-4, device: str = 'cuda'):
    """Train diffusion model (VQ-VAE frozen)."""
    # Freeze VQ-VAE
    for param in model.vqvae.parameters():
        param.requires_grad = False

    # optimizer = torch.optim.Adam(model.denoiser.parameters(), lr=lr)
    model.to(device)
    model.load_clip(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (voxels, text_emb) in enumerate(dataloader):
            voxels = voxels.to(device)
            text_emb = text_emb.to(device)

            optimizer.zero_grad()
            loss, metrics = model.compute_diffusion_loss(voxels, text_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}, Avg Loss: {total_loss / len(dataloader):.3f}")


# ============================================================================
# Example Usage
# ============================================================================


class VoxelDataset(Dataset):
    def __init__(self, voxel_grids: list[torch.Tensor], clip_embeddings: torch.Tensor):
        self.data = torch.tensor(np.array(voxel_grids))  # [N, 16, 16, 16]
        self.clip_embeddings = clip_embeddings  # [N, LABEL_EMBED_DIMENSIONS]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.clip_embeddings[index]


def train_diffusion_vqvae_model():
    VOCAB_SIZE = get_max_block_id() + 1  # Number of unique texture IDs
    NUM_EMBEDDINGS = 512  # Codebook size
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = VoxelDiffusionModel(
        vocab_size=VOCAB_SIZE,
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=256,
        d_model=512,
        n_layers=6,
        n_heads=8,
        num_timesteps=1000
    ).to(DEVICE)

    lr=1e-4
    optimizer_vqvae = torch.optim.Adam(model.vqvae.parameters(), lr=lr)
    optimizer_denoiser = torch.optim.Adam(model.denoiser.parameters(), lr=lr)

    # load clip
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    clip_cache = {}

    def get_clip_embedding(text_label: str) -> torch.Tensor:
        if text_label not in clip_cache:
            tokens = clip.tokenize([text_label]).to(DEVICE)
            with torch.no_grad():
                clip_cache[text_label] = clip_model.encode_text(tokens).squeeze(0).float()
        return clip_cache[text_label]

    # load data
    text_labels, voxel_data_np = get_random_training_dataset(2048)
    with torch.no_grad():
        clip_embeddings = torch.stack([get_clip_embedding(label) for label in text_labels])

    dataset = VoxelDataset(voxel_data_np, clip_embeddings)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    num_epochs = 100
    print("Train VQVAE")
    train_vqvae(model, optimizer_vqvae, dataloader, num_epochs=num_epochs, device=DEVICE)
    print("Train Diffusion")
    train_diffusion(model, optimizer_denoiser, dataloader, num_epochs=num_epochs, device=DEVICE)

    # Save checkpoint
    torch.save({
        'epoch': num_epochs,
        'model': model.state_dict(),
        'optimizer_vqvae': optimizer_vqvae.state_dict(),
        'optimizer_denoiser': optimizer_denoiser.state_dict(),
        'clip_cache': clip_cache,
    }, f"data/model/diffusion-vqvae-checkpoint-{num_epochs}.pth")
    print("Checkpoint saved!")


def load_model(
        file_path: str = "data/model/diffusion-vqvae-checkpoint-100.pth",
        lr: float = 1e-4):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    VOCAB_SIZE = get_max_block_id() + 1  # Number of unique texture IDs
    NUM_EMBEDDINGS = 512  # Codebook size

    # load checkpoint
    checkpoint = torch.load(file_path)
    epoch = checkpoint['epoch']

    print(f'Loading diffusion model (epoch {epoch})')

    # load model
    model = VoxelDiffusionModel(
        vocab_size=VOCAB_SIZE,
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=256,
        d_model=512,
        n_layers=6,
        n_heads=8,
        num_timesteps=1000
    ).to(DEVICE)
    model.load_clip(DEVICE)
    model.load_state_dict(checkpoint['model'])

    # load optimisers
    optimizer_vqvae = torch.optim.Adam(model.vqvae.parameters(), lr=lr)
    optimizer_vqvae.load_state_dict(checkpoint["optimizer_vqvae"])
    optimizer_denoiser = torch.optim.Adam(model.denoiser.parameters(), lr=lr)
    optimizer_denoiser.load_state_dict(checkpoint["optimizer_denoiser"])

    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)

    return model, optimizer_vqvae, optimizer_denoiser, clip_model, epoch

@torch.no_grad()
def sample_diffusion_model(
        prompt: str,
        guidance_scale: float = 2.0,
        num_samples: int = 1,
        steps: int = 50) -> np.ndarray:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, optimizer_vqvae, optimizer_denoiser, clip_model, epoch = load_model()
    model.eval()
    clip_model.eval()

    # Encode prompt with CLIP
    tokens = clip.tokenize([prompt]).to(device)
    clip_emb = clip_model.encode_text(tokens).float()  # (1, 512)

    # Expand for multiple samples
    clip_emb = clip_emb.repeat(num_samples, 1)  # (num_samples, 512)

    # For classifier-free guidance, we need unconditional embeddings
    uncond_emb = torch.zeros_like(clip_emb)  # Null embedding

    # Initialize with all mask tokens
    B = num_samples
    indices = torch.full((B, 4, 4, 4), model.mask_token_id,
                         dtype=torch.long, device=device)

    # Create timestep schedule (subsample if steps < num_timesteps)
    total_timesteps = model.diffusion.num_timesteps
    timestep_schedule = torch.linspace(total_timesteps - 1, 0, steps, dtype=torch.long)

    # Reverse diffusion
    for step_idx, t in enumerate(timestep_schedule):
        print(f"Sampling step {step_idx + 1}/{steps} (t={t})")

        t_batch = torch.full((B,), t, dtype=torch.float, device=device)

        if guidance_scale != 1.0:
            # Classifier-free guidance
            # Concatenate conditional and unconditional
            indices_input = torch.cat([indices, indices], dim=0)
            t_input = torch.cat([t_batch, t_batch], dim=0)
            emb_input = torch.cat([clip_emb, uncond_emb], dim=0)

            # Get predictions
            logits = model.denoiser(indices_input, t_input, emb_input)

            # Split conditional and unconditional
            logits_cond, logits_uncond = logits.chunk(2, dim=0)

            # Apply guidance
            logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
        else:
            # No guidance
            logits = model.denoiser(indices, t_batch, clip_emb)

        # Sample from categorical distribution
        probs = F.softmax(logits, dim=-1)  # (B, 64, num_embeddings+1)

        # Sample for each position
        sampled = torch.multinomial(
            probs.view(-1, model.num_embeddings + 1),
            num_samples=1
        ).view(B, 4, 4, 4)

        # Keep non-mask tokens (absorbing property)
        # Only update positions that are still masked
        mask = indices == model.mask_token_id
        indices = torch.where(mask, sampled, indices)

        # Optional: For faster convergence, use confidence-based unmasking
        # Only unmask the most confident predictions
        if step_idx < len(timestep_schedule) - 1:
            # Calculate confidence (max probability)
            confidence = probs.max(dim=-1)[0]  # (B, 64)
            confidence = confidence.view(B, 4, 4, 4)

            # Calculate how many tokens to keep masked for next step
            num_masked = mask.sum().item()
            next_t = timestep_schedule[step_idx + 1]
            alpha_bar_next = model.diffusion.alphas_cumprod[next_t]
            target_masked = int((1 - alpha_bar_next) * 64 * B)

            if num_masked > target_masked and target_masked > 0:
                # Re-mask the least confident predictions
                num_to_remask = num_masked - target_masked
                flat_confidence = confidence.view(-1)
                flat_indices = indices.view(-1)
                flat_mask = mask.view(-1)

                # Find least confident unmasked tokens
                unmasked_positions = ~flat_mask
                if unmasked_positions.any():
                    unmasked_confidence = flat_confidence.clone()
                    unmasked_confidence[flat_mask] = float('inf')  # Ignore already masked

                    _, lowest_conf_idx = torch.topk(unmasked_confidence,
                                                    k=min(num_to_remask, unmasked_positions.sum().item()),
                                                    largest=False)
                    flat_indices[lowest_conf_idx] = model.mask_token_id
                    indices = flat_indices.view(B, 4, 4, 4)

    # Decode to voxels
    print("Decoding to voxels...")
    voxel_logits = model.vqvae.decode_indices(indices)  # (B, vocab_size, 16, 16, 16)
    voxels = voxel_logits.argmax(dim=1)  # (B, 16, 16, 16)

    # Convert to numpy
    voxels_np = voxels.cpu().numpy()

    return voxels_np