import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm
import torch.optim as optim
import torch.nn.utils as utils
from pathlib import Path

from scripts.transformations.randomize_data import get_random_dataset


class Generator3D(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 16 * 4 * 4 * 4),
            nn.ReLU(True),

            nn.Unflatten(1, (16, 4, 4, 4)),  # 4x4x4
            nn.BatchNorm3d(16),
            nn.ReLU(True),

            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),  # 8x8x8
            nn.BatchNorm3d(8),
            nn.ReLU(True),

            nn.ConvTranspose3d(8, 1, kernel_size=4, stride=2, padding=1),  # 16x16x16
            nn.Sigmoid()
        )

        # 200
        """
        nn.Linear(latent_dim, 512 * 4 * 4 * 4),
        
        nn.Unflatten(1, (512, 4, 4, 4)),  # 4x4x4
        nn.BatchNorm3d(512),
        nn.ReLU(True),
        
        nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),  # 8x8x8
        nn.BatchNorm3d(256),
        nn.ReLU(True),
        
        nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16x16
        nn.BatchNorm3d(128),
        nn.ReLU(True),
        
        nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32x32
        nn.BatchNorm3d(64),
        nn.ReLU(True),
        
        nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),  # 64x64x64
        nn.Sigmoid()
        """

    def forward(self, z):
        return self.model(z)


class Discriminator3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv3d(1, 8, kernel_size=4, stride=2, padding=1)),  # 8x8x8
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv3d(8, 16, kernel_size=4, stride=2, padding=1)),  # 4x4x4
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            spectral_norm(nn.Linear(16 * 4 * 4 * 4, 1)),
        )

        """
        nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1),  # 32x32x32
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16x16
        nn.BatchNorm3d(128),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8x8
        nn.BatchNorm3d(256),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),  # 4x4x4
        nn.BatchNorm3d(512),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Flatten(),
        nn.Linear(512 * 4 * 4 * 4, 1),
        """

    def forward(self, x):
        return self.model(x)


def d_loss(real_scores, fake_scores):
    return fake_scores.mean() - real_scores.mean()


def g_loss(fake_scores):
    return -fake_scores.mean()


def gradient_penalty(D, real, fake, device='cpu'):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, 1, device=device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    d_interpolated = D(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty


class VoxelDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data).unsqueeze(1)  # (N, 1, 16, 16, 16)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_gan(generator=None, discriminator=None, g_opt=None, d_opt=None, last_epoch=0, latent_dim=128, epochs=100, batch_size=64,
              lambda_gp=10, critic_iters=3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if generator is None:
        generator = Generator3D
    if discriminator is None:
        discriminator = Discriminator3D
    if g_opt is None:
        g_opt = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    if d_opt is None:
        d_opt = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    generator.to(device)
    discriminator.to(device)

    for epoch in range(last_epoch, last_epoch + epochs):
        real_data_np = get_random_dataset(512)
        dataset = VoxelDataset(real_data_np)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for i, real in enumerate(dataloader):
            real = real.to(device)
            batch_size_curr = real.size(0)

            # real_labels = torch.full((batch_size_curr, 1), 0.7).to(device)  # reduce to avoid discriminator overpowering generator
            # fake_labels = torch.zeros((batch_size_curr, 1)).to(device)

            # --- Train Discriminator ---

            # train discriminator multiple times
            for _ in range(critic_iters):
                z = torch.randn(batch_size_curr, latent_dim).to(device)
                fake = generator(z).detach()
                real_scores = discriminator(real)
                fake_scores = discriminator(fake)

                gp = gradient_penalty(discriminator, real, fake, device=device)
                d_loss_val = d_loss(real_scores, fake_scores) + lambda_gp * gp

                # d_real_loss = criterion(discriminator(real), real_labels)
                # d_fake_loss = criterion(discriminator(fake.detach()), fake_labels)
                # d_loss = d_real_loss + d_fake_loss

                d_opt.zero_grad()
                d_loss_val.backward()
                d_opt.step()

            # --- Train Generator ---
            z = torch.randn(batch_size_curr, latent_dim).to(device)
            fake = generator(z)
            fake_scores = discriminator(fake)
            g_loss_val = g_loss(fake_scores)

            g_opt.zero_grad()
            g_loss_val.backward()
            g_opt.step()

        print(f"Epoch {epoch + 1}/{last_epoch + epochs} | D Loss: {d_loss_val.item():.2f} | G Loss: {g_loss_val.item():.2f} | GP: {gp.item():.2f}")

        if epoch > 0 and epoch % 50 == 0:
            save_model(generator, discriminator, g_opt, d_opt, epoch)

    save_model(generator, discriminator, g_opt, d_opt, epoch + 1)


def continue_training_gan():
    generator, discriminator, g_opt, d_opt, epoch = load_model()
    generator.train()
    discriminator.train()
    train_gan(generator, discriminator, g_opt, d_opt, epoch)


def save_model(generator, discriminator, g_opt, d_opt, epoch):
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'g_optimizer': g_opt.state_dict(),
        'd_optimizer': d_opt.state_dict(),
        'epoch': epoch
    }, f"data/model/checkpoint-{epoch}.pth")
    print("Checkpoint saved!")


def load_model(file_path="data/model/checkpoint-200.pth"):
    checkpoint = torch.load(file_path)
    generator = Generator3D()
    generator.load_state_dict(checkpoint['generator'])
    discriminator = Discriminator3D()
    discriminator.load_state_dict(checkpoint['discriminator'])
    g_opt = optim.Adam(generator.parameters(), lr=5e-4, betas=(0.5, 0.9))
    g_opt.load_state_dict(checkpoint['g_optimizer'])
    d_opt = optim.Adam(discriminator.parameters(), lr=5e-4, betas=(0.5, 0.9))
    d_opt.load_state_dict(checkpoint['d_optimizer'])
    epoch = checkpoint['epoch']
    return generator, discriminator, g_opt, d_opt, epoch


def generate_voxel(generator=None, latent_dim=128, device='cpu'):
    if generator is None:
        generator, discriminator, g_opt, d_opt, epoch = load_model()
    generator.eval()
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        voxel = generator(z)
        binary = (voxel > 0.5).int().squeeze().cpu().numpy()  # Shape: (16, 16, 16)
    return binary
