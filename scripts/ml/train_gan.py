import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm
import torch.optim as optim
from scripts.transformations.randomize_data import get_random_dataset


class ConditionalBatchNorm3d(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.bn = nn.BatchNorm3d(num_features, affine=False)
        self.gamma_embed = nn.Linear(embed_dim, num_features)
        self.beta_embed = nn.Linear(embed_dim, num_features)

    def forward(self, x, label_emb):
        out = self.bn(x)
        gamma = self.gamma_embed(label_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_embed(label_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma * out + beta


class Generator3D(nn.Module):
    def __init__(self, latent_dim=128, num_classes=6, embed_dim=64):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        input_dim = latent_dim + embed_dim

        self.latent_to_tensor = nn.Sequential(
            nn.Linear(input_dim, 128 * 2 * 2 * 2),
            nn.ReLU(True)
        )

        # Instead of one big Sequential, break up the blocks to allow label input
        self.deconv1 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)  # 4x4x4
        self.cbn1 = ConditionalBatchNorm3d(64, embed_dim)

        self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)   # 8x8x8
        self.cbn2 = ConditionalBatchNorm3d(32, embed_dim)

        self.deconv3 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1)   # 16x16x16
        self.cbn3 = ConditionalBatchNorm3d(16, embed_dim)

        self.to_voxel = nn.Sequential(
            nn.Conv3d(19, 1, kernel_size=3, padding=1),  # 16x16x16 + 3 for pos encoding
            nn.Sigmoid()
        )

    def forward(self, z, labels):
        label_embedding = self.label_emb(labels)  # (B, embed_dim)
        z = torch.cat((z, label_embedding), dim=1)

        x = self.latent_to_tensor(z)
        x = x.view(-1, 128, 2, 2, 2)  # Unflatten

        x = self.deconv1(x)
        x = self.cbn1(x, label_embedding)
        x = torch.relu(x)

        x = self.deconv2(x)
        x = self.cbn2(x, label_embedding)
        x = torch.relu(x)

        x = self.deconv3(x)
        x = self.cbn3(x, label_embedding)
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
    def __init__(self, num_classes=6, embed_dim=64):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        self.conv_layers = nn.Sequential(
            spectral_norm(nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1)),  # 8x8x8
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1)),  # 4x4x4
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)),  # 2x2x2
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = spectral_norm(nn.Linear(64 * 2 * 2 * 2, 1))
        self.embed_proj = nn.Linear(embed_dim, 64 * 2 * 2 * 2)

    def forward(self, x, labels):
        batch_size = x.size(0)
        features = self.conv_layers(x).view(batch_size, -1)  # shape: (B, F)
        out = self.fc(features).squeeze(1)                   # scalar output: (B,)

        # Projection term
        label_embedding = self.label_emb(labels)             # (B, embed_dim)
        projection = torch.sum(self.embed_proj(label_embedding) * features, dim=1)  # (B,)

        return out + projection


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
    def __init__(self, voxel_data, label_list, label_to_index):
        self.data = torch.tensor(voxel_data).unsqueeze(1)  # (N, 1, 16, 16, 16)
        self.labels = torch.tensor([label_to_index[label] for label in label_list], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


label_to_index = {
    "hollow cuboid": 0,
    "solid cuboid": 1,
    "hollow pyramid": 2,
    "solid pyramid": 3,
    "hollow sphere": 4,
    "solid sphere": 5,
}


def train_gan(generator=None, discriminator=None, g_opt=None, d_opt=None, last_epoch=0, latent_dim=256, epochs=800, batch_size=64,
              lambda_gp=10, critic_iters=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if generator is None:
        generator = Generator3D(latent_dim=latent_dim)
    if discriminator is None:
        discriminator = Discriminator3D()
    if g_opt is None:
        g_opt = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    if d_opt is None:
        d_opt = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    generator.to(device)
    discriminator.to(device)

    for epoch in range(last_epoch, last_epoch + epochs):
        input_labels, voxel_data_np = get_random_dataset(512)
        dataset = VoxelDataset(voxel_data_np, input_labels, label_to_index)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for i, (real, labels) in enumerate(dataloader):
            real = real.to(device)
            labels = labels.to(device)
            batch_size_curr = real.size(0)

            # train discriminator multiple times
            for _ in range(critic_iters):
                z = torch.randn(batch_size_curr, latent_dim).to(device)
                fake = generator(z, labels).detach()
                real_scores = discriminator(real, labels)
                fake_scores = discriminator(fake, labels)

                gp = gradient_penalty(lambda r: discriminator(r, labels), real, fake, device=device)
                d_loss_val = d_loss(real_scores, fake_scores) + lambda_gp * gp

                d_opt.zero_grad()
                d_loss_val.backward()
                d_opt.step()

            # train generator
            z = torch.randn(batch_size_curr, latent_dim).to(device)
            fake = generator(z, labels)
            fake_scores = discriminator(fake, labels)
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
    }, f"data/model/gan-checkpoint-{epoch}.pth")
    print("Checkpoint saved!")


def load_model(file_path="data/model/gan-checkpoint-200.pth"):
    checkpoint = torch.load(file_path)
    generator = Generator3D(latent_dim=256)
    generator.load_state_dict(checkpoint['generator'])
    discriminator = Discriminator3D()
    discriminator.load_state_dict(checkpoint['discriminator'])
    g_opt = optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5, 0.9))
    g_opt.load_state_dict(checkpoint['g_optimizer'])
    d_opt = optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.9))
    d_opt.load_state_dict(checkpoint['d_optimizer'])
    epoch = checkpoint['epoch']
    print(f'Loading WGAN-SN (epoch {epoch})')
    return generator, discriminator, g_opt, d_opt, epoch


def sample_gan(input_label, generator=None, latent_dim=256, device='cpu'):
    if generator is None:
        generator, discriminator, g_opt, d_opt, epoch = load_model()
    generator.eval()
    label_idx = torch.tensor([label_to_index[input_label]], device=device)  # shape: [1]

    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        voxel = generator(z, label_idx)
        binary = (voxel > 0.5).int().squeeze().cpu().numpy()  # Shape: (16, 16, 16)
    return binary
