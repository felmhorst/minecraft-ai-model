import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from scripts.transformations.randomize_data import get_random_dataset


class VoxelDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data).unsqueeze(1)  # Shape: [N, 1, 16, 16, 16]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Simple3DUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU()
        )
        self.middle = nn.Sequential(
            nn.Conv3d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv3d(16, 1, 3, padding=1)
        )

    def forward(self, x, t):
        # Add timestep embedding
        t_embed = t[:, None, None, None, None].float()
        x = x + t_embed
        x = self.encoder(x)
        x = self.middle(x)
        return self.decoder(x)


class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
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
                        x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * noise_pred) + torch.sqrt(beta_t) * noise
        return x


def train_diffusion_model(model=None, optimizer=None, last_epoch=0, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = Simple3DUNet()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    data = get_random_dataset(512)
    dataset = VoxelDataset(data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model.to(device)
    diffusion = Diffusion(timesteps=200)

    for epoch in range(last_epoch, last_epoch + epochs):
        for x in tqdm(dataloader):
            x = x.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device)
            x_noisy, noise = diffusion.noise_input(x, t)
            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{last_epoch + epochs} | Loss: {loss.item():.2f}")

    save_model(model, optimizer, epoch + 1)


def continue_training_diffusion_model():
    model, optimizer, epoch = load_model()
    model.train()
    train_diffusion_model(model, optimizer, epoch)


def save_model(model, optimizer, epoch):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }, f"data/model/diffusion-checkpoint-{epoch}.pth")
    print("Checkpoint saved!")


def load_model(file_path="data/model/diffusion-checkpoint-10.pth"):
    checkpoint = torch.load(file_path)
    model = Simple3DUNet()
    model.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def sample_diffusion_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, epoch = load_model()
    model.eval()
    diffusion = Diffusion(timesteps=200)
    with torch.no_grad():
        samples = diffusion.sample(model, (1, 1, 16, 16, 16), device)
        sample = samples[0, 0].clamp(0, 1).cpu().numpy()
        binary_voxels = (sample > 0.5).astype(np.uint8)
    return binary_voxels
