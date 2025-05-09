import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scripts.transformations.randomize_data import get_random_dataset

base_path = Path(__file__).parent
model_file = base_path / '..' / '..' / 'data' / 'model' / 'model.pth'


# Custom Dataset for voxel data
class VoxelDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data).unsqueeze(1)  # (N, 1, 16, 16, 16)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 3D VAE Model
class VoxelVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=2, padding=1),  # -> (8, 8, 8, 8)
            nn.ReLU(),
            nn.Conv3d(8, 16, 3, stride=2, padding=1), # -> (16, 4, 4, 4)
            nn.ReLU(),
            nn.Flatten()
        )
        self.enc_out_dim = 16 * 4 * 4 * 4
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.enc_out_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16, 4, 4, 4)),
            nn.ConvTranspose3d(16, 8, 4, stride=2, padding=1),  # (8, 8, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose3d(8, 1, 4, stride=2, padding=1),   # (1, 16, 16, 16)
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar):
    # Binary cross-entropy loss for reconstruction
    bce = nn.functional.binary_cross_entropy(recon, x, reduction='sum')
    # KL divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kl


def train(model=VoxelVAE(), epochs=1000, batch_size=32, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        data = get_random_dataset(512)
        dataset = VoxelDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.2f}")

    torch.save(model.state_dict(), model_file)
    print(f"VAE model saved to {model_file}")