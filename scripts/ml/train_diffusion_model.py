import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from scripts.transformations.randomize_data import get_random_dataset


# ==== Dataset ====
class VoxelDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data).unsqueeze(1).float()  # [N, 1, 16, 16, 16]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ==== Autoencoder ====
class VoxelAutoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()

        self.latent_dim = latent_dim
        # latent_vector_dim = 128

        self.encoder_cnn = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),  # 32x8x8x8
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),    # 64x4x4x4
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),  # 128x4x4x4
            nn.BatchNorm3d(128),
            nn.LeakyReLU()

            # nn.Conv3d(64, latent_dim, kernel_size=3, stride=1, padding=1),  # latent_dimx4x4x4

            # nn.Flatten(),
            # nn.Linear(latent_dim * 2 * 2 * 2, latent_vector_dim)
        )

        self.conv_mu = nn.Conv3d(128, latent_dim, kernel_size=3, padding=1)
        self.conv_logvar = nn.Conv3d(128, latent_dim, kernel_size=3, padding=1)

        self.decoder = nn.Sequential(
            # nn.Linear(latent_vector_dim, latent_dim * 2 * 2 * 2),
            # nn.Unflatten(1, (latent_dim, 2, 2, 2)),

            nn.ConvTranspose3d(latent_dim, 64, kernel_size=4, stride=2, padding=1),  # 64x8x8x8
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x16x16x16
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),

            nn.Conv3d(32, 1, kernel_size=3, padding=1),  # 1x16x16x16
            nn.Sigmoid()
        )

    def encode(self, x):
        # return self.encoder(x)
        h = self.encoder_cnn(x)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


# ==== UNet 3D ====
class UNet3D(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_channels = latent_dim + 3

        self.enc1 = nn.Sequential(
            nn.Conv3d(self.input_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.middle = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        self.dec1 = nn.Sequential(
            nn.Conv3d(64 + 64, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv3d(32 + 32, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.output_layer = nn.Conv3d(16, latent_dim, 3, padding=1)  # Predict noise in latent space

    def forward(self, x, t):
        t_embed = t[:, None, None, None, None].float()
        x = x + t_embed
        x = self.add_positional_encoding(x)

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x_mid = self.middle(x2)

        x = self.dec1(torch.cat([x_mid, x2], dim=1))
        x = self.dec2(torch.cat([x, x1], dim=1))
        return self.output_layer(x)

    def add_positional_encoding(self, voxel_input):
        B, C, D, H, W = voxel_input.shape
        device = voxel_input.device
        z = torch.linspace(-1, 1, D, device=device).view(1, 1, D, 1, 1).expand(B, 1, D, H, W)
        y = torch.linspace(-1, 1, H, device=device).view(1, 1, 1, H, 1).expand(B, 1, D, H, W)
        x = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, 1, W).expand(B, 1, D, H, W)
        return torch.cat([voxel_input, x, y, z], dim=1)


# ==== Diffusion ====
class Diffusion:
    def __init__(self, timesteps=200, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_input(self, x0, t):
        noise = torch.randn_like(x0)
        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1, 1).to(x0.device)
        return torch.sqrt(alpha_hat_t) * x0 + torch.sqrt(1 - alpha_hat_t) * noise, noise

    def sample(self, model, shape, device):
        x = torch.randn(shape).to(device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((shape[0],), t, dtype=torch.long).to(device)
            noise_pred = model(x, t_tensor)
            beta_t = self.beta[t].to(device)
            alpha_t = self.alpha[t].to(device)
            alpha_hat_t = self.alpha_hat[t].to(device)

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            x = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * noise_pred
            ) + torch.sqrt(beta_t) * noise
        return x


def kl_divergence(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


# ==== Training ====
def train_latent_diffusion_model(latent_dim=16, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = VoxelAutoencoder(latent_dim=latent_dim).to(device)
    diffusion_model = UNet3D(latent_dim=latent_dim).to(device)

    optimizer = torch.optim.Adam(
        list(autoencoder.parameters()) + list(diffusion_model.parameters()), lr=1e-4
    )

    data = get_random_dataset(4096)
    dataset = VoxelDataset(data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    diffusion = Diffusion(timesteps=100)

    for epoch in range(epochs):
        autoencoder.train()
        diffusion_model.train()
        for x in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device)

            # z = autoencoder.encode(x) # (B, latent_dim, 4, 4, 4)
            mu, logvar = autoencoder.encode(x)
            z = autoencoder.reparameterize(mu, logvar)
            z_noisy, noise = diffusion.noise_input(z, t)
            noise_pred = diffusion_model(z_noisy, t)

            high_weight = 1.0 + 8.0 * (epoch / epochs) ** 2

            importance_mask = x.clamp(min=0.0, max=1.0)
            importance_mask_latent = F.interpolate(importance_mask, size=z.shape[2:], mode='trilinear', align_corners=False)
            weights = 1.0 + importance_mask_latent * high_weight
            recon_loss = (weights * (noise_pred - noise) ** 2).mean()

            # KL divergence loss
            kl_loss = kl_divergence(mu, logvar)

            total_loss = recon_loss + 1e-4 * kl_loss  # Adjust KL weight

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} | Recon Loss: {recon_loss.item():.2f} | KL Loss: {kl_loss.item():.2f} | Total: {total_loss.item():.2f}")

    save_model(autoencoder, diffusion_model, optimizer, epoch + 1)


# ==== Save/Load ====
def save_model(autoencoder, model, optimizer, epoch):
    torch.save({
        'autoencoder': autoencoder.state_dict(),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }, f"data/model/latent-diffusion-checkpoint-{epoch}.pth")
    print("Checkpoint saved!")


def load_model(file_path="data/model/latent-diffusion-gan-checkpoint-100.pth", latent_dim=16):
    checkpoint = torch.load(file_path)
    autoencoder = VoxelAutoencoder(latent_dim)
    model = UNet3D(latent_dim)
    optimizer = torch.optim.Adam(list(autoencoder.parameters()) + list(model.parameters()), lr=1e-4)

    autoencoder.load_state_dict(checkpoint['autoencoder'])
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint (epoch {epoch})")
    return autoencoder, model, optimizer, epoch


# ==== Sampling ====
def sample_latent_diffusion(latent_dim=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder, model, _, _ = load_model(latent_dim=latent_dim)
    autoencoder.to(device).eval()
    model.to(device).eval()
    diffusion = Diffusion(timesteps=50)

    with torch.no_grad():
        z = diffusion.sample(model, (1, latent_dim, 4, 4, 4), device)
        x_recon = autoencoder.decode(z)
        voxels = x_recon[0, 0].clamp(0, 1).cpu().numpy()
        voxels_int = (voxels > .5).astype(np.uint8)
        # voxels_int = np.clip((voxels * 5), 0, 4).astype(np.uint8)
        return voxels_int
