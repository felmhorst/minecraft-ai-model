import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm
import torch.optim as optim
import clip

from scripts.conversion.occupancy_udf_conversion import occupancy_to_udf, udf_to_occupancy
from scripts.training_data.get_random_training_dataset import get_random_training_dataset
from scripts.training_data.normalize_block_ids import get_max_block_id
from scripts.visualize.visualize_voxel_grid import visualize_voxel_grid
import time

MODEL_NAME = "WGAN-GP"
LABEL_EMBED_DIMENSIONS: int = 512  # CLIP ViT-B/32 output size
NUM_TEXTURES: int = get_max_block_id() + 1  # 9
TEXTURE_EMBED_DIMENSIONS: int = 4

TEMPERATURE: float = 1.0  # smooth logits during softmax


class FiLM3D(nn.Module):
    """feature-wise linear modulation"""
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.gamma = nn.Linear(embed_dim, in_channels)
        self.beta = nn.Linear(embed_dim, in_channels)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W], y: [B, embed_dim]
        gamma = self.gamma(y).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(y).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class ConditionalBatchNorm3d(nn.Module):
    """a variant of BatchNorm3d that includes a label embedding"""
    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm3d(num_features, affine=False)
        self.gamma_embed = nn.Linear(embed_dim, num_features)
        self.beta_embed = nn.Linear(embed_dim, num_features)

    def forward(self, x: torch.Tensor, label_emb: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        gamma = self.gamma_embed(label_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_embed(label_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma * out + beta


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False, upsample: bool = False):
        super().__init__()
        assert not (downsample and upsample), "Can't both downsample and upsample in the same ResBlock"
        self.downsample = downsample
        self.upsample = upsample

        self.conv1 = spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv2 = spectral_norm(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))

        # learned 1x1 conv in skip when channel counts differ or sampling occurs
        if in_channels != out_channels or downsample or upsample:
            self.skip = spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(x)
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode="trilinear", align_corners=False)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool3d(h, kernel_size=2, stride=2)

        # skip path
        skip = x
        if self.upsample:
            skip = F.interpolate(skip, scale_factor=2, mode="trilinear", align_corners=False)
        if self.downsample:
            skip = F.avg_pool3d(skip, kernel_size=2, stride=2)
        if hasattr(self, "skip"):
            skip = self.skip(skip)

        return h + skip


class Generator3D(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        input_dimensions = latent_dim + LABEL_EMBED_DIMENSIONS

        # Latent projection -> small 3D tensor (coarse)
        self.latent_to_tensor = nn.Sequential(
            nn.Linear(input_dimensions, 256 * 2 * 2 * 2),
            nn.ReLU(True)
        )

        # Encoder (coarse -> bottleneck)
        self.enc1 = ResBlock3D(256, 256, downsample=False)    # [B,256,2,2,2]
        self.enc2 = ResBlock3D(256, 128, downsample=True)     # [B,128,1,1,1]

        # Decoder with multi-scale skips (we will reuse enc1 upsampled to multiple sizes)
        self.dec1 = ResBlock3D(128, 128, upsample=True)       # [B,128,2,2,2]
        self.film1 = FiLM3D(128, LABEL_EMBED_DIMENSIONS)

        # dec2 expects concat(d1, enc1_up_to_d1) -> channels = 128 + 256 = 384
        self.dec2 = ResBlock3D(128 + 256, 128, upsample=True) # [B,128,4,4,4]
        self.film2 = FiLM3D(128, LABEL_EMBED_DIMENSIONS)

        # dec3 expects concat(d2, enc1_up_to_d2) -> channels = 128 + 256 = 384
        self.dec3 = ResBlock3D(128 + 256, 64, upsample=True)  # [B,64,8,8,8]
        self.film3 = FiLM3D(64, LABEL_EMBED_DIMENSIONS)

        # dec4 expects concat(d3, enc1_up_to_d3) -> channels = 64 + 256 = 320
        self.dec4 = ResBlock3D(64 + 256, 32, upsample=True)   # [B,32,16,16,16]
        self.film4 = FiLM3D(32, LABEL_EMBED_DIMENSIONS)

        # Heads (add positional enc -> occupancy + texture)
        channels_with_positional_encoding = 32 + 3
        self.to_occupancy = nn.Conv3d(channels_with_positional_encoding, 1, kernel_size=3, padding=1)
        self.to_texture = nn.Conv3d(channels_with_positional_encoding, NUM_TEXTURES, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, label_embeddings: torch.Tensor):
        # z: [B, latent_dim], label_embeddings: [B, LABEL_EMBED_DIMENSIONS]
        z = torch.cat([z, label_embeddings], dim=1)
        x = self.latent_to_tensor(z).reshape(-1, 256, 2, 2, 2)  # [B,256,2,2,2]

        # ---- Encoder ----
        e1 = self.enc1(x)   # [B,256,2,2,2]
        e2 = self.enc2(e1)  # [B,128,1,1,1] bottleneck

        # ---- Decoder ----
        d1 = self.dec1(e2)            # [B,128,2,2,2]
        d1 = self.film1(d1, label_embeddings)

        # skip1 = e1 aligned to d1 spatial size
        skip1 = e1
        if skip1.shape[2:] != d1.shape[2:]:
            skip1 = F.interpolate(skip1, size=d1.shape[2:], mode="trilinear", align_corners=False)

        d2_in = torch.cat([d1, skip1], dim=1)   # [B, 128+256, 2,2,2] -> dec2 expects this
        d2 = self.dec2(d2_in)                   # [B,128,4,4,4]
        d2 = self.film2(d2, label_embeddings)

        # skip2 = e1 aligned to d2 spatial size
        skip2 = e1
        if skip2.shape[2:] != d2.shape[2:]:
            skip2 = F.interpolate(skip2, size=d2.shape[2:], mode="trilinear", align_corners=False)

        d3_in = torch.cat([d2, skip2], dim=1)   # [B, 128 + 256, 4,4,4]
        d3 = self.dec3(d3_in)                   # [B,64,8,8,8]
        d3 = self.film3(d3, label_embeddings)

        # skip3 = e1 aligned to d3 spatial size
        skip3 = e1
        if skip3.shape[2:] != d3.shape[2:]:
            skip3 = F.interpolate(skip3, size=d3.shape[2:], mode="trilinear", align_corners=False)

        d4_in = torch.cat([d3, skip3], dim=1)   # [B, 64 + 256, 8,8,8]
        d4 = self.dec4(d4_in)                   # [B,32,16,16,16]
        d4 = self.film4(d4, label_embeddings)

        # ---- Positional encoding + heads ----
        B, _, D, H, W = d4.shape
        pos = self.get_positional_grid(D, d4.device).unsqueeze(0).repeat(B, 1, 1, 1, 1)
        feat = torch.cat([d4, pos], dim=1)  # [B, 32+3, 16,16,16]

        occupancy_logits = self.to_occupancy(feat)
        texture_logits   = self.to_texture(feat)
        return occupancy_logits, texture_logits

    def sample_texture_ids(self, z: torch.Tensor, label_embeddings: torch.Tensor) -> torch.Tensor:
        occupancy_logits, texture_logits = self.forward(z, label_embeddings)
        occupancy = udf_to_occupancy(occupancy_logits)

        # Optionally include texture here:
        # texture = torch.argmax(texture_logits, dim=1, keepdim=True)
        # voxel_ids = occupancy * torch.clamp(texture, min=1)

        return occupancy.int()

    @staticmethod
    def get_positional_grid(size, device):
        ranges = [torch.linspace(-1, 1, steps=size, device=device) for _ in range(3)]
        grid_z, grid_y, grid_x = torch.meshgrid(ranges, indexing="ij")
        return torch.stack([grid_x, grid_y, grid_z], dim=0)


class Discriminator3D(nn.Module):
    def __init__(self):
        super().__init__()

        # max_norm caps the embedding
        self.texture_embedding = nn.Embedding(NUM_TEXTURES, TEXTURE_EMBED_DIMENSIONS, max_norm=1.0)

        # in_channels = TEXTURE_EMBED_DIMENSIONS + 1  # texture embedding + occupancy
        in_channels = 1  # texture embedding + occupancy

        self.conv_layers = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1)),  # [batch_size, 32, 8, 8, 8]
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)),  # [batch_size, 64, 4, 4, 4]
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)),  # [batch_size, 128, 2, 2, 2]
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv3d(128, 256, kernel_size=2, stride=1, padding=0)),  # [batch_size, 256, 1, 1, 1]
            nn.LeakyReLU(0.2, inplace=True),
        )

        # flatten final features
        self.fc = spectral_norm(nn.Linear(256, 1))

        self.embed_proj = nn.Linear(LABEL_EMBED_DIMENSIONS, 256)  # [batch_size, 512]

    def forward(
            self,
            occupancy: torch.Tensor,
            texture_logits: torch.Tensor,
            label_embeddings: torch.Tensor,
            already_embedded: bool = False) -> torch.Tensor:
        """
        Calculates the Wasserstein critic values for a batch of samples.
        :param occupancy: tensor of shape [batch_size, 1, 16, 16, 16]
        :param texture_logits: tensor of shape [batch_size, TEXTURE_EMBED_DIMENSIONS, 16, 16, 16] (if already_embedded)
        :param label_embeddings: tensor of shape [batch_size, LABEL_EMBED_DIMENSIONS]
        :param already_embedded: boolean
        :return: tensor of shape [batch_size], where each entry is the Wasserstein critic value for that sample
        """
        # embed texture_id to [TEXTURE_EMBED_DIMENSIONS]
        if not already_embedded:
            texture_logits = self.texture_embedding(texture_logits)  # [batch_size, 16, 16, 16, TEXTURE_EMBED_DIMENSIONS]
            texture_logits = texture_logits.permute(0, 4, 1, 2, 3)   # [batch_size, TEXTURE_EMBED_DIMENSIONS, 16, 16, 16]

        # combine occupancy and texture
        # x = torch.cat([occupancy.float(), texture_logits], dim=1)
        x = occupancy.float()
        batch_size = x.size(0)
        features = self.conv_layers(x).reshape(batch_size, -1)  # [batch_size, 256]

        out = self.fc(features).squeeze(1)  # [batch_size]

        # evaluate label consistency
        projection = torch.sum(self.embed_proj(label_embeddings) * features, dim=1)  # [batch_size]

        return out + projection  # [batch_size]

    def forward_combined_embeddings(
            self,
            x: torch.Tensor,
            label_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass when inputs are already concatenated occupancy+texture embeddings.
        :param x: [batch_size, 1+TEXTURE_EMBED_DIMENSIONS, 16, 16, 16]
        :param label_embeddings: [batch_size, LABEL_EMBED_DIMENSIONS]
        :return: tensor of shape [batch_size], where each entry is the Wasserstein critic value for that sample
        """
        batch_size = x.size(0)
        features = self.conv_layers(x).reshape(batch_size, -1)  # [batch_size, 512]

        out = self.fc(features).squeeze(1)  # [batch_size]
        projection = torch.sum(self.embed_proj(label_embeddings) * features, dim=1)  # [batch_size]

        return out + projection  # [batch_size]


def calculate_discriminator_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Calculates the discriminator's loss. In a WGAN, the discriminator should maximize the difference between the two
    scores by increasing real_scores and decreasing fake_scores.
    :param real_scores: scores that the discriminator calculated for real samples, tensor of shape [batch_size]
    :param fake_scores: scores that the discriminator calculated for samples from generator, tensor of
        shape [batch_size]
    :returns: discriminator loss of shape [] (a single scalar)
    """
    return fake_scores.mean() - real_scores.mean()


def calculate_generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Calculates the generator's loss. In a WGAN, the generator should maximize the fake_scores.
    :param fake_scores: scores that the discriminator calculated for samples generated by the generator,
        tensor of shape [batch_size]
    :returns: generator loss of shape [] (a single scalar)
    """
    return -fake_scores.mean()


def calculate_gradient_penalty(
        discriminator: Discriminator3D,
        real_occupancy: torch.Tensor,
        real_textures: torch.Tensor,
        fake_occupancy_logits: torch.Tensor,
        fake_texture_logits: torch.Tensor,
        label_embeddings: torch.Tensor,
        device: str | torch.device = 'cpu',
) -> torch.Tensor:
    """
    Calculates the gradient penalty.
    :param discriminator: Discriminator3D
    :param real_occupancy: batch of real occupancies, tensor of shape [batch_size, 1, 16, 16, 16]
    :param real_textures: batch of real texture ids, tensor of shape [batch_size, 16, 16, 16]
    :param fake_occupancy_logits: batch of occupancy logits from generator, tensor of shape [batch_size, 1, 16, 16, 16]
    :param fake_texture_logits: batch of texture logits from generator, tensor of shape
        [batch_size, NUM_TEXTURES, 16, 16, 16]
    :param label_embeddings: tensor of shape [batch_size, LABEL_EMBED_DIMENSIONS]
    :param device: cpu or cuda
    :return: gradient penalty of shape [] (a single scalar)
    """

    # get a random scalar per sample in the batch
    batch_size = real_occupancy.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, 1, device=device)  # [batch_size, 1, 1, 1, 1]

    # embed textures (real batch)
    # real_texture_embed = discriminator.texture_embedding(real_textures).permute(0, 4, 1, 2, 3)  # [batch_size, TEXTURE_EMBED_DIMENSIONS, 16, 16, 16]

    # embed textures (generated batch)
    # fake_texture_probs = torch.softmax(fake_texture_logits / TEMPERATURE, dim=1)  # [batch_size, NUM_TEXTURES, 16, 16, 16] (ratios)
    # fake_texture_embed = torch.einsum("bndhw,ne->bedhw", fake_texture_probs, discriminator.texture_embedding.weight)  # [batch_size, TEXTURE_EMBED_DIMENSIONS, 16, 16, 16]

    fake_occupancy_probs = torch.sigmoid(fake_occupancy_logits).float()  # [batch_size, 1, 16, 16, 16]
    real_occupancy_float = real_occupancy.float()  # [batch_size, 1, 16, 16, 16]

    # combine occupancy + texture
    # real_x = torch.cat([real_occupancy_float, real_texture_embed], dim=1)
    # fake_x = torch.cat([fake_occupancy_probs, fake_texture_embed], dim=1)
    real_x = real_occupancy_float
    fake_x = fake_occupancy_probs

    # interpolate gradients between real and fake
    interpolated = epsilon * real_x + (1 - epsilon) * fake_x  # [batch_size, TEXTURE_EMBED_DIMENSIONS, 16, 16, 16]
    interpolated.requires_grad_(True)

    # calculate critic value
    d_interpolated = discriminator.forward_combined_embeddings(interpolated, label_embeddings)  # [batch_size]

    # compute gradients for every sample
    # gradients has shape [batch_size, 1+TEXTURE_EMBED_DIMENSIONS, 16, 16, 16]
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # # compute per-voxel squared norms
    # gradients_sq = gradients.pow(2)  # [batch_size, 1+TEXTURE_EMBED_DIM, 16, 16, 16]
    #
    # # compute squared occupancy gradient
    # occupancy_per_voxel_sq = gradients_sq[:, 0, ...]  # [batch_size, 16, 16, 16]
    # occupancy_gradient_sq = occupancy_per_voxel_sq.sum(dim=[1, 2, 3])  # [batch_size]
    #
    # # computed weighted squared texture gradient
    # # the texture's gradient depends on the occupancy. an occupancy closer to 1 is more heavily penalized
    # texture_per_voxel_sq = gradients_sq[:, 1:, ...].sum(dim=1)  # [batch_size, 16, 16, 16]
    # occupancy_interpolated = interpolated[:, 0, ...]  # [batch_size, 16, 16, 16]
    # weighted_texture_sq = texture_per_voxel_sq * occupancy_interpolated  # [batch_size, 16, 16, 16]
    # texture_gradient_sq = weighted_texture_sq.sum(dim=[1, 2, 3])  # [batch_size]
    #
    # # combine gradients + compute penalty
    # gradient_norm = torch.sqrt(occupancy_gradient_sq + texture_gradient_sq + 1e-12)
    # gradient_penalty = ((gradient_norm - 1) ** 2).mean()  # [] (single scalar)

    # Norm over spatial + channel dims (just occupancy channel here)
    gradients = gradients.view(batch_size, -1)  # [B, 1*16*16*16]
    gradient_norm = gradients.norm(2, dim=1)  # [B]

    # WGAN-GP penalty
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


class VoxelDataset(Dataset):
    def __init__(self, voxel_grids: list, clip_embeddings: torch.Tensor):
        self.data = torch.tensor(np.array(voxel_grids)).long()  # [batch_size, 16, 16, 16]
        self.clip_embeddings = clip_embeddings  # [batch_size, LABEL_EMBED_DIMENSIONS]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        voxel_grid = self.data[idx]  # [16, 16, 16]
        occupancy_grid = (voxel_grid > 0).long().unsqueeze(0)  # [1, 16, 16, 16]
        texture_grid = voxel_grid.long()
        return occupancy_grid, texture_grid, self.clip_embeddings[idx]


def train_gan(
        generator: Generator3D = None,
        discriminator: Discriminator3D = None,
        generator_optimiser: optim.Adam | optim.RMSprop = None,
        discriminator_optimiser: optim.Adam | optim.RMSprop = None,
        last_clip_cache=None,
        last_epoch: int = 0,
        latent_dim: int = 256,
        epochs: int = 100,
        batch_size: int = 64,
        lambda_gp: float = 10,
        discriminator_iterations: int = 5,
        generator_iterations: int = 1,
        lr: float = 1e-4,
        alpha_adversarial: float = 1.0,
        alpha_occupancy: float = 0.0,
        alpha_texture: float = 0.0
):
    """trains a GAN"""
    start_time = time.time()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if generator is None:
        generator = Generator3D(latent_dim=latent_dim)
    if discriminator is None:
        discriminator = Discriminator3D()
    if generator_optimiser is None:
        # generator_optimiser = optim.RMSprop(generator.parameters(), lr=lr, alpha=0.9)
        generator_optimiser = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    if discriminator_optimiser is None:
        # discriminator_optimiser = optim.RMSprop(discriminator.parameters(), lr=lr, alpha=0.9)
        discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    # update learning rate
    generator_optimiser.param_groups[0]['lr'] = lr
    discriminator_optimiser.param_groups[0]['lr'] = lr

    clip_cache = last_clip_cache if last_clip_cache is not None else {}

    generator.to(device)
    discriminator.to(device)

    # initialize clip model
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # get clip cache
    def get_clip_embedding(text_label: str) -> torch.Tensor:
        """
        returns the cached clip embedding for the given text_label or calculates it.
        :param text_label: the text to embed
        :return: the text's embedding
        """
        if text_label not in clip_cache:
            tokens = clip.tokenize([text_label]).to(device)
            with torch.no_grad():
                clip_cache[text_label] = clip_model.encode_text(tokens).squeeze(0).to(device).float()
        return clip_cache[text_label]

    # epoch
    for epoch in range(last_epoch, last_epoch + epochs):
        # prepare training data
        text_labels, voxel_data_np = get_random_training_dataset(512)
        with torch.no_grad():
            clip_embs = torch.stack([get_clip_embedding(lbl) for lbl in text_labels]).to(device)
        dataset = VoxelDataset(voxel_data_np, clip_embs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # batch
        for i, (real_occupancy, real_textures, clip_embs) in enumerate(dataloader):
            # real_occupancy = real_occupancy.to(device).long()
            real_occupancy = occupancy_to_udf(real_occupancy).to(device)
            real_textures = real_textures.to(device).long()
            clip_embs = clip_embs.to(device)
            batch_size_current = real_occupancy.size(0)

            # train discriminator
            for _ in range(discriminator_iterations):
                z = torch.randn(batch_size_current, latent_dim).to(device)
                occupancy_logits, texture_logits = generator(z, clip_embs)
                occupancy_logits = occupancy_logits.detach()
                texture_logits = texture_logits.detach()

                # sigmoid is applied so the occupancy is in the same space as the real occupancy (0-1) when passing it
                # to the discriminator
                occupancy_probabilities = torch.sigmoid(occupancy_logits)

                # fake_occupancy = (occupancy_probabilities > 0.5).long()

                # real_embed and fake_embed are handled inside gradient_penalty
                real_scores = discriminator(real_occupancy, real_textures, clip_embs)  # IDs → embedded internally
                fake_texture_probs = torch.softmax(texture_logits / TEMPERATURE, dim=1)
                fake_texture_embed = torch.einsum("bndhw,ne->bedhw", fake_texture_probs, discriminator.texture_embedding.weight)
                fake_scores = discriminator(occupancy_probabilities, fake_texture_embed, clip_embs, already_embedded=True)

                gp = calculate_gradient_penalty(discriminator, real_occupancy, real_textures, occupancy_logits, texture_logits, clip_embs, device=device)
                discriminator_loss = calculate_discriminator_loss(real_scores, fake_scores) + lambda_gp * gp

                discriminator_optimiser.zero_grad()
                discriminator_loss.backward()
                discriminator_optimiser.step()

            # train generator
            for _ in range(generator_iterations):
                z = torch.randn(batch_size_current, latent_dim).to(device)
                occupancy_logits, texture_logits = generator(z, clip_embs)
                occupancy_probabilities = torch.sigmoid(occupancy_logits)

                fake_texture_probs = torch.softmax(texture_logits / TEMPERATURE, dim=1)
                fake_texture_embed = torch.einsum("bndhw,ne->bedhw", fake_texture_probs, discriminator.texture_embedding.weight)

                fake_scores = discriminator(occupancy_probabilities, fake_texture_embed, clip_embs, already_embedded=True)

                # calculate occupancy loss
                # occupancy_loss = nn.functional.binary_cross_entropy_with_logits(occupancy_logits, real_occupancy.float())

                l1_loss_fn = nn.L1Loss()
                occupancy_loss = l1_loss_fn(occupancy_probabilities, real_occupancy.float())

                # calculate texture loss (where occupancy > 0)
                mask = (real_textures > 0)  # [batch_size, 16, 16, 16]
                if mask.any():
                    texture_logits = texture_logits.permute(0, 2, 3, 4, 1)  # [batch_size, 16, 16, 16, NUM_TEXTURES]
                    texture_loss = nn.functional.cross_entropy(texture_logits[mask], real_textures[mask])
                else:
                    texture_loss = torch.tensor(0.0, device=device)

                # calculate generator loss
                adversarial_loss = calculate_generator_loss(fake_scores)

                # combine losses
                generator_loss = alpha_adversarial * adversarial_loss + alpha_occupancy * occupancy_loss + alpha_texture * texture_loss

                generator_optimiser.zero_grad()
                generator_loss.backward()
                generator_optimiser.step()

        print(f"Epoch {epoch + 1}/{last_epoch + epochs} | D Loss: {discriminator_loss.item():.2f} | G Loss: {generator_loss.item():.2f} | GP: {gp.item():.2f}")

        if epoch > 0 and epoch % 100 == 0:
            labels = ["gable house", "steep gable roof house", "a-frame house", "desert house", "solid pyramid", "hollow cuboid"]
            clip_embeddings = []
            for label in labels:
                clip_embeddings.append(get_clip_embedding(label))
            test_model(generator, clip_embeddings, labels, latent_dim, device)
        if epoch > 0 and epoch % 100 == 0:
            save_model(generator, discriminator, generator_optimiser, discriminator_optimiser, clip_cache, epoch)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Finished training for {epochs} epochs ({duration:.2f} seconds)")
    save_model(generator, discriminator, generator_optimiser, discriminator_optimiser, clip_cache, epoch + 1)


def test_model(
        generator: Generator3D,
        clip_embeddings: list[torch.Tensor],
        labels: list[str],
        latent_dim: int,
        device: torch.device | str
):
    generator.eval()
    voxel_grids = []
    with torch.no_grad():
        z = torch.randn(1, latent_dim, device=device)
        for clip_embedding in clip_embeddings:
            clip_embedding.to(device)
            if clip_embedding.dim() == 1:
                clip_embedding = clip_embedding.unsqueeze(0)  # -> [1, LABEL_EMBED_DIMENSIONS]
            elif clip_embedding.dim() == 2 and clip_embedding.size(0) != 1:
                # if user passed a batch >1, optionally handle it; here we sample only the first
                clip_embedding = clip_embedding[:1, :]

            voxel_data = generator.sample_texture_ids(z, clip_embedding)
            if voxel_data.dim() == 5 and voxel_data.size(1) == 1:
                voxel_data = voxel_data.squeeze(1)

            data_np = voxel_data.squeeze(0).cpu().numpy()
            voxel_grids.append(data_np)
    visualize_voxel_grid(voxel_grids, labels)
    generator.train()


def continue_training_gan(
        last_epoch: int = 100,
        epochs: int = 100,
        lr: float = 1e-4,
        alpha_adversarial: float = 1.0,
        alpha_occupancy: float = 0.0,
        alpha_texture: float = 0.0
):
    generator, discriminator, g_opt, d_opt, clip_cache, epoch = load_model(file_path=f"data/model/gan-checkpoint-{last_epoch}.pth", lr=lr)
    generator.train()
    discriminator.train()
    train_gan(generator, discriminator, g_opt, d_opt, clip_cache, epoch, epochs=epochs, lr=lr,
              alpha_adversarial=alpha_adversarial, alpha_occupancy=alpha_occupancy, alpha_texture=alpha_texture)


def save_model(
        generator: Generator3D,
        discriminator: Discriminator3D,
        generator_optimiser: optim.Adam | optim.RMSprop,
        discriminator_optimiser: optim.Adam | optim.RMSprop,
        clip_cache,
        epoch: int
):
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'g_optimizer': generator_optimiser.state_dict(),
        'd_optimizer': discriminator_optimiser.state_dict(),
        'clip_cache': clip_cache,
        'epoch': epoch
    }, f"data/model/gan-checkpoint-{epoch}.pth")
    print("Checkpoint saved!")


def load_model(file_path: str = "data/model/gan-checkpoint-400.pth", lr=1e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load checkpoint
    checkpoint = torch.load(file_path)
    epoch = checkpoint['epoch']
    clip_cache = checkpoint['clip_cache']
    print(f'Loading {MODEL_NAME} (epoch {epoch})')

    # initialize generator
    generator = Generator3D(latent_dim=256).to(device)
    generator.load_state_dict(checkpoint['generator'])

    # initialize discriminator
    discriminator = Discriminator3D().to(device)
    discriminator.load_state_dict(checkpoint['discriminator'])

    # initialize generator optimiser
    g_opt = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    # g_opt = optim.RMSprop(generator.parameters(), lr=lr, alpha=0.9)
    g_opt.load_state_dict(checkpoint['g_optimizer'])

    # initialize discriminator optimiser
    d_opt = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
    # d_opt = optim.RMSprop(discriminator.parameters(), lr=lr, alpha=0.9)
    d_opt.load_state_dict(checkpoint['d_optimizer'])

    return generator, discriminator, g_opt, d_opt, clip_cache, epoch


def sample_gan(
        input_label: str,
        generator: Generator3D = None,
        latent_dim: int = 256):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if generator is None:
        generator, discriminator, g_opt, d_opt, clip_cache, epoch = load_model()
    generator.to(device)
    generator.eval()

    # text processing
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    with torch.no_grad():
        text_feat = clip_model.encode_text(clip.tokenize([input_label]).to(device)).float()
    label_emb = text_feat.to(device)

    # generate voxel data
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        voxel_data = generator.sample_texture_ids(z, label_emb)

        data_np = voxel_data.squeeze().cpu().numpy()

        # binary = (voxel > 0.5).int().squeeze().cpu().numpy()  # Shape: (16, 16, 16)
    return data_np


def train_gan_by_schedule():
    train_gan(epochs=100, lr=2e-4, alpha_adversarial=0.5, alpha_occupancy=1.0, alpha_texture=0.0)
    continue_training_gan(last_epoch=100, epochs=100, lr=2e-4, alpha_adversarial=1.0, alpha_occupancy=0.6, alpha_texture=0.0)
    continue_training_gan(last_epoch=200, epochs=100, lr=1e-4, alpha_adversarial=1.0, alpha_occupancy=0.3, alpha_texture=0.0)
    continue_training_gan(last_epoch=300, epochs=100, lr=1e-5, alpha_adversarial=1.0, alpha_occupancy=0.1, alpha_texture=0.0)
