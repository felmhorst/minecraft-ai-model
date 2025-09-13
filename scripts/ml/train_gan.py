import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import clip
from scripts.training_data.get_random_training_dataset import get_random_training_dataset
from scripts.training_data.normalize_block_ids import get_max_block_id, denormalize_block_ids

MODEL_NAME = "WGAN-GP"
LABEL_EMBED_DIMENSIONS = 512  # CLIP ViT-B/32
NUM_TEXTURES = get_max_block_id() + 1
TEXTURE_EMBED_DIMENSIONS = 4


class ConditionalBatchNorm3d(nn.Module):
    """a variant of BatchNorm3d that includes a label embedding"""
    def __init__(self, num_features, embed_dimensions):
        super().__init__()
        self.bn = nn.BatchNorm3d(num_features, affine=False)
        self.gamma_embed = nn.Linear(embed_dimensions, num_features)
        self.beta_embed = nn.Linear(embed_dimensions, num_features)

    def forward(self, x, label_embed):
        out = self.bn(x)
        gamma = self.gamma_embed(label_embed).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_embed(label_embed).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma * out + beta


class Generator3D(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        input_dim = latent_dim + LABEL_EMBED_DIMENSIONS

        self.latent_to_tensor = nn.Sequential(
            nn.Linear(input_dim, 128 * 2 * 2 * 2),
            nn.ReLU(True)
        )

        # Instead of one big Sequential, break up the blocks to allow label input
        self.deconv1 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)  # 4x4x4
        self.cbn1 = ConditionalBatchNorm3d(64, LABEL_EMBED_DIMENSIONS)

        self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)   # 8x8x8
        self.cbn2 = ConditionalBatchNorm3d(32, LABEL_EMBED_DIMENSIONS)

        self.deconv3 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1)   # 16x16x16
        self.cbn3 = ConditionalBatchNorm3d(16, LABEL_EMBED_DIMENSIONS)

        self.to_voxel = nn.Conv3d(16 + 3, 1, kernel_size=3, padding=1)  # 16x16x16 + 3 for pos encoding

    def forward(self, z, label_embeddings):
        z = torch.cat((z, label_embeddings), dim=1)

        x = self.latent_to_tensor(z)
        x = x.view(-1, 128, 2, 2, 2)  # Unflatten

        x = self.deconv1(x)
        x = self.cbn1(x, label_embeddings)
        x = torch.relu(x)

        x = self.deconv2(x)
        x = self.cbn2(x, label_embeddings)
        x = torch.relu(x)

        x = self.deconv3(x)
        x = self.cbn3(x, label_embeddings)
        x = torch.relu(x)

        # Add positional encoding
        B, _, D, H, W = x.shape
        device = x.device
        pos = self.get_positional_grid(D, device).unsqueeze(0).repeat(B, 1, 1, 1, 1)
        x = torch.cat([x, pos], dim=1)

        return self.to_voxel(x)

    @staticmethod
    def get_positional_grid(size, device):
        ranges = [torch.linspace(-1, 1, steps=size, device=device) for _ in range(3)]
        grid_z, grid_y, grid_x = torch.meshgrid(ranges, indexing='ij')
        return torch.stack([grid_x, grid_y, grid_z], dim=0)


class Discriminator3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),  # 8x8x8, 16 channel
            # nn.InstanceNorm3d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),  # 4x4x4, 32 channel
            # nn.InstanceNorm3d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),  # 2x2x2, 64 channel
            # nn.InstanceNorm3d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Linear(64 * 2 * 2 * 2, 1)
        self.embed_proj = nn.Linear(LABEL_EMBED_DIMENSIONS, 64 * 2 * 2 * 2)

    def forward(self, x, label_embeddings):
        batch_size = x.size(0)
        features = self.conv_layers(x).view(batch_size, -1)  # shape: (B, F)
        out = self.fc(features).squeeze(1)                   # scalar output: (B,)

        projection = torch.sum(self.embed_proj(label_embeddings) * features, dim=1)  # (B,)

        return out + projection


def discriminator_loss(real_scores, fake_scores):
    return fake_scores.mean() - real_scores.mean()


def generator_loss(fake_scores):
    return -fake_scores.mean()


def gradient_penalty(discriminator, real, fake, label_embs, device='cpu'):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, 1, device=device)

    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    # old
    d_interpolated = discriminator(interpolated, label_embs)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # new
    # d_interpolated = discriminator.conv_layers(interpolated).reshape(batch_size, -1)
    # d_interpolated_out = discriminator.fc(d_interpolated).squeeze(1)
    # projection = torch.sum(discriminator.embed_proj(label_embs) * d_interpolated, dim=1)
    # d_interpolated_combined = d_interpolated_out + projection
    # gradients = torch.autograd.grad(
    #     outputs=d_interpolated_combined,
    #     inputs=interpolated,
    #     grad_outputs=torch.ones_like(d_interpolated_combined),
    #     create_graph=True,
    #     retain_graph=True,
    #     only_inputs=True
    # )[0]

    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty


class VoxelDataset(Dataset):
    def __init__(self, voxel_data, clip_embs):
        self.data = torch.tensor(np.array(voxel_data)).unsqueeze(1).float()  # (BATCH, 1, 16, 16, 16)
        self.clip_embs = clip_embs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.clip_embs[idx]


def train_gan(generator=None,
              discriminator=None,
              generator_optimiser=None,
              discriminator_optimiser=None,
              last_clip_cache=None,
              last_epoch=0,
              latent_dim=256,
              epochs=100,
              batch_size=64,
              lambda_gp=10,
              critic_iterations=5,
              generator_iterations=1):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if generator is None:
        generator = Generator3D(latent_dim=latent_dim)
    if discriminator is None:
        discriminator = Discriminator3D()
    if generator_optimiser is None:
        generator_optimiser = optim.RMSprop(generator.parameters(), lr=1e-5, alpha=0.99)
        # generator_optimiser = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    if discriminator_optimiser is None:
        discriminator_optimiser = optim.RMSprop(discriminator.parameters(), lr=1e-4, alpha=0.99)
        # discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9))

    clip_cache = last_clip_cache if last_clip_cache is not None else {}

    generator.to(device)
    discriminator.to(device)

    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # CLIP CACHE
    def get_clip_embedding(text_label):
        if text_label not in clip_cache:
            tokens = clip.tokenize([text_label]).to(device)
            with torch.no_grad():
                clip_cache[text_label] = clip_model.encode_text(tokens).squeeze(0).to(device).float()
        return clip_cache[text_label]

    # epoch
    for epoch in range(last_epoch, last_epoch + epochs):
        input_labels, voxel_data_np = get_random_training_dataset(512)
        with torch.no_grad():
            clip_embs = torch.stack([get_clip_embedding(lbl) for lbl in input_labels]).to(device)
        dataset = VoxelDataset(voxel_data_np, clip_embs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # batch
        for i, (real, clip_embs) in enumerate(dataloader):
            real = real.to(device)
            clip_embs = clip_embs.to(device)
            batch_size_curr = real.size(0)

            # train discriminator
            for _ in range(critic_iterations):
                z = torch.randn(batch_size_curr, latent_dim).to(device)
                fake = generator(z, clip_embs).detach()
                real_scores = discriminator(real, clip_embs)
                fake_scores = discriminator(fake, clip_embs)

                gp = gradient_penalty(discriminator, real, fake, clip_embs, device=device)
                d_loss_val = discriminator_loss(real_scores, fake_scores) + lambda_gp * gp

                discriminator_optimiser.zero_grad()
                d_loss_val.backward()
                discriminator_optimiser.step()

            # train generator
            for _ in range(generator_iterations):
                z = torch.randn(batch_size_curr, latent_dim).to(device)
                fake = generator(z, clip_embs)

                fake_scores = discriminator(fake, clip_embs)
                g_loss_val = generator_loss(fake_scores)

                generator_optimiser.zero_grad()
                g_loss_val.backward()
                generator_optimiser.step()

        print(f"Epoch {epoch + 1}/{last_epoch + epochs} | D Loss: {d_loss_val.item():.2f} | G Loss: {g_loss_val.item():.2f} | GP: {gp.item():.2f}")

        if epoch > 0 and epoch % 100 == 0:
            save_model(generator, discriminator, generator_optimiser, discriminator_optimiser, clip_cache, epoch)

    save_model(generator, discriminator, generator_optimiser, discriminator_optimiser, clip_cache, epoch + 1)


def continue_training_gan():
    generator, discriminator, g_opt, d_opt, clip_cache, epoch = load_model()
    generator.train()
    discriminator.train()
    train_gan(generator, discriminator, g_opt, d_opt, clip_cache, epoch)


def save_model(generator, discriminator, g_opt, d_opt, clip_cache, epoch):
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'g_optimizer': g_opt.state_dict(),
        'd_optimizer': d_opt.state_dict(),
        'clip_cache': clip_cache,
        'epoch': epoch
    }, f"data/model/gan-checkpoint-{epoch}.pth")
    print("Checkpoint saved!")


def load_model(file_path="data/model/gan-checkpoint-100.pth"):
    checkpoint = torch.load(file_path)
    generator = Generator3D(latent_dim=256)
    generator.load_state_dict(checkpoint['generator'])
    discriminator = Discriminator3D()
    discriminator.load_state_dict(checkpoint['discriminator'])

    g_opt = optim.RMSprop(generator.parameters(), lr=1e-5, alpha=0.99)
    # g_opt = optim.Adam(generator.parameters(), lr=2e-5, betas=(0.0, 0.9))
    g_opt.load_state_dict(checkpoint['g_optimizer'])

    d_opt = optim.RMSprop(discriminator.parameters(), lr=1e-4, alpha=0.99)
    # d_opt = optim.Adam(discriminator.parameters(), lr=2e-5, betas=(0.0, 0.9))
    d_opt.load_state_dict(checkpoint['d_optimizer'])

    clip_cache = checkpoint['clip_cache']
    epoch = checkpoint['epoch']
    print(f'Loading WGAN-SN (epoch {epoch})')
    return generator, discriminator, g_opt, d_opt, clip_cache, epoch


def sample_gan(input_label, generator=None, latent_dim=256,):
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

    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        voxel = generator(z, label_emb)
        data_np = voxel.squeeze().cpu().numpy()

    # clamp to 0-1
    data_np = np.clip(data_np, 0.0, 1.0)

    # denormalize block ids to 0-max_texture_id
    data_3d = denormalize_block_ids(data_np)
    return data_3d
