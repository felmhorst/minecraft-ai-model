import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm
import torch.optim as optim
import clip
from scripts.get_random_training_dataset import get_random_training_dataset
from scripts.normalize_block_ids import get_max_block_id

MODEL_NAME = "WGAN-GP"
LABEL_EMBED_DIMENSIONS: int = 512  # CLIP ViT-B/32 output size
NUM_TEXTURES: int = get_max_block_id() + 1  # 9
TEXTURE_EMBED_DIMENSIONS: int = 4

TEMPERATURE: float = 2.0  # smooth logits during softmax


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


class Generator3D(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()

        input_dim = latent_dim + LABEL_EMBED_DIMENSIONS

        self.latent_to_tensor = nn.Sequential(
            nn.Linear(input_dim, 128 * 2 * 2 * 2),
            nn.ReLU(True)
        )

        self.deconv1 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.cbn1 = ConditionalBatchNorm3d(64, LABEL_EMBED_DIMENSIONS)

        self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.cbn2 = ConditionalBatchNorm3d(32, LABEL_EMBED_DIMENSIONS)

        self.deconv3 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1)
        self.cbn3 = ConditionalBatchNorm3d(16, LABEL_EMBED_DIMENSIONS)

        # +3 for positional encoding
        self.to_voxel = nn.Conv3d(16 + 3, NUM_TEXTURES, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, label_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculates the logits (unnormalized scores) per texture in a batch of noisy voxel grids, based on a label.
        :param z: random noise, a tensor of shape [batch_size, latent_dim]
        :param label_embeddings: embedded labels, a tensor of shape [batch_size, LABEL_EMBED_DIMENSIONS]
        :return: likelihood per texture, a tensor of shape [batch_size, NUM_TEXTURES, 16, 16, 16]
        """
        # z: [batch_size, latent_dim]
        z = torch.cat((z, label_embeddings), dim=1)  # [batch_size, latent_dim+LABEL_EMBED_DIMENSIONS]

        x = self.latent_to_tensor(z)
        x = x.view(-1, 128, 2, 2, 2)  # [batch_size, 128, 2, 2, 2]

        x = self.deconv1(x)  # [batch_size, 64, 4, 4, 4]
        x = self.cbn1(x, label_embeddings)
        x = torch.relu(x)

        x = self.deconv2(x)  # [batch_size, 32, 8, 8, 8]
        x = self.cbn2(x, label_embeddings)
        x = torch.relu(x)

        x = self.deconv3(x)  # [batch_size, 16, 16, 16, 16]
        x = self.cbn3(x, label_embeddings)
        x = torch.relu(x)

        # add positional encoding
        B, _, D, H, W = x.shape
        device = x.device
        pos = self.get_positional_grid(D, device).unsqueeze(0).repeat(B, 1, 1, 1, 1)
        x = torch.cat([x, pos], dim=1)  # [batch_size, 19, 16, 16, 16] (added 3 channels for positional encoding)

        return self.to_voxel(x)  # [batch_size, NUM_TEXTURES, 16, 16, 16] (likelihood per texture)

    def sample_texture_ids(self, z: torch.Tensor, label_embeddings: torch.Tensor) -> torch.Tensor:
        logits = self.forward(z, label_embeddings)
        return torch.argmax(logits, dim=1)  # [batch_size, 16, 16, 16] (texture ids)

    @staticmethod
    def get_positional_grid(size: int, device: str) -> torch.Tensor:
        """returns a tensor of shape [3, size, size, size] that represents a 3D positional grid of normalized
        coordinates in the range [-1, 1]"""
        ranges = [torch.linspace(-1, 1, steps=size, device=device) for _ in range(3)]  # 3 x [size] in range [-1, 1]
        grid_z, grid_y, grid_x = torch.meshgrid(ranges, indexing='ij')  # 3 x [size, size]
        return torch.stack([grid_x, grid_y, grid_z], dim=0)  # [3, size, size, size]


class Discriminator3D(nn.Module):
    def __init__(self):
        super().__init__()

        # max_norm caps the embedding
        self.texture_embedding = nn.Embedding(NUM_TEXTURES, TEXTURE_EMBED_DIMENSIONS, max_norm=1.0)

        self.conv_layers = nn.Sequential(
            nn.Conv3d(TEXTURE_EMBED_DIMENSIONS, 16, kernel_size=4, stride=2, padding=1),  # [batch_size, 16, 8, 8, 8]
            # spectral_norm(nn.Conv3d(TEXTURE_EMBED_DIMENSIONS, 16, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),  # [batch_size, 32, 4, 4, 4]
            # spectral_norm(nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),  # [batch_size, 64, 2, 2, 2]
            # spectral_norm(nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Linear(64 * 2 * 2 * 2, 1)  # [batch_size, 1]
        # self.fc = spectral_norm(nn.Linear(64 * 2 * 2 * 2, 1))

        self.embed_proj = nn.Linear(LABEL_EMBED_DIMENSIONS, 64 * 2 * 2 * 2)  # [batch_size, 512]

    def forward(self, x: torch.Tensor, label_embeddings: torch.Tensor, already_embedded: bool = False):
        """
        Calculates the Wasserstein critic values for a batch of samples.
        :param x: tensor of shape [batch_size, TEXTURE_EMBED_DIMENSIONS, 16, 16, 16] (if already_embedded)
        :param label_embeddings: tensor of shape [batch_size, LABEL_EMBED_DIMENSIONS]
        :param already_embedded: boolean
        :return: tensor of shape [batch_size], where each entry is the Wasserstein critic value for that sample
        """

        # embed texture_id to [TEXTURE_EMBED_DIMENSIONS]
        if not already_embedded:
            x = self.texture_embedding(x)  # [batch_size, 16, 16, 16, TEXTURE_EMBED_DIMENSIONS]
            x = x.permute(0, 4, 1, 2, 3)   # [batch_size, TEXTURE_EMBED_DIMENSIONS, 16, 16, 16]

        batch_size = x.size(0)
        features = self.conv_layers(x).reshape(batch_size, -1)  # [batch_size, 512]

        out = self.fc(features).squeeze(1)  # [batch_size]

        # evaluate label consistency
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
        real: torch.Tensor,
        fake_logits: torch.Tensor,
        label_embeddings: torch.Tensor,
        device: str = 'cpu') -> torch.Tensor:
    """
    Calculates the gradient penalty.
    :param discriminator: Discriminator3D
    :param real: batch of real samples, tensor of shape [batch_size, 16, 16, 16]
    :param fake_logits: batch of logits from generator, tensor of shape [batch_size, NUM_TEXTURES, 16, 16, 16]
    :param label_embeddings: tensor of shape [batch_size, LABEL_EMBED_DIMENSIONS]
    :param device: cpu or cuda
    :return: gradient penalty of shape [] (a single scalar)
    """
    # get a random scalar per sample in the batch
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, 1, device=device)  # [batch_size, 1, 1, 1, 1]

    # get embedded representations
    # fake_embed = discriminator.texture_embedding(fake)  # (B, D, H, W, C)
    # real_embed = discriminator.texture_embedding(real)  # (B, D, H, W, C)
    # real_embed = real_embed.permute(0, 4, 1, 2, 3)
    # fake_embed = fake_embed.permute(0, 4, 1, 2, 3)

    # embed textures (real batch)
    real_embed = discriminator.texture_embedding(real).permute(0, 4, 1, 2, 3)  # [batch_size, TEXTURE_EMBED_DIMENSIONS, 16, 16, 16]

    # embed textures (generated batch)
    fake_probs = torch.softmax(fake_logits / TEMPERATURE, dim=1)  # [batch_size, NUM_TEXTURES, 16, 16, 16] (ratios)
    fake_embed = torch.einsum("bndhw,ne->bedhw", fake_probs, discriminator.texture_embedding.weight)  # [batch_size, TEXTURE_EMBED_DIMENSIONS, 16, 16, 16]

    # interpolate gradients between real and fake
    interpolated = epsilon * real_embed + (1 - epsilon) * fake_embed  # [batch_size, TEXTURE_EMBED_DIMENSIONS, 16, 16, 16]
    interpolated.requires_grad_(True)

    # calculate critic value
    d_interpolated = discriminator(interpolated, label_embeddings, already_embedded=True)  # [batch_size]

    # compute gradients for every sample
    # gradients has shape [batch_size, TEXTURE_EMBED_DIMENSIONS, 16, 16, 16]
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # compute gradient penalty
    gradients = gradients.reshape(batch_size, -1)  # [batch_size, TEXTURE_EMBED_DIMENSIONS*16*16*16]
    gradients_norm = gradients.norm(2, dim=1)  # [batch_size]
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()  # [] (single scalar)
    return gradient_penalty


class VoxelDataset(Dataset):
    def __init__(self, voxel_data: torch.Tensor, clip_embs: torch.Tensor):
        self.data = torch.tensor(np.array(voxel_data)).long()  # [batch_size, TEXTURE_EMBED_DIMENSIONS, 16, 16, 16]
        self.clip_embs = clip_embs  # [batch_size, LABEL_EMBED_DIMENSIONS]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.clip_embs[idx]


def train_gan(
        generator: Generator3D = None,
        discriminator: Discriminator3D = None,
        generator_optimiser: optim.Adam | optim.RMSprop = None,
        discriminator_optimiser: optim.Adam | optim.RMSprop = None,
        last_clip_cache=None,
        last_epoch: int = 0,
        latent_dim: int = 256,
        epochs: int = 50,
        batch_size: int = 64,
        lambda_gp: float = 10,
        discriminator_iterations: int = 5,
        generator_iterations: int = 1):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if generator is None:
        generator = Generator3D(latent_dim=latent_dim)
    if discriminator is None:
        discriminator = Discriminator3D()
    if generator_optimiser is None:
        # generator_optimiser = optim.RMSprop(generator.parameters(), lr=1e-4, alpha=0.9)
        generator_optimiser = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    if discriminator_optimiser is None:
        # discriminator_optimiser = optim.RMSprop(discriminator.parameters(), lr=1e-4, alpha=0.9)
        discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

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
        for i, (real, clip_embs) in enumerate(dataloader):
            real = real.to(device).long()
            clip_embs = clip_embs.to(device)
            batch_size_current = real.size(0)

            # train discriminator
            for _ in range(discriminator_iterations):
                z = torch.randn(batch_size_current, latent_dim).to(device)
                fake_logits = generator(z, clip_embs).detach()

                # real_embed and fake_embed are handled inside gradient_penalty
                real_scores = discriminator(real, clip_embs)  # IDs → embedded internally
                fake_probs = torch.softmax(fake_logits / TEMPERATURE, dim=1)
                fake_embed = torch.einsum("bndhw,ne->bedhw", fake_probs, discriminator.texture_embedding.weight)
                fake_scores = discriminator(fake_embed, clip_embs, already_embedded=True)

                gp = calculate_gradient_penalty(discriminator, real, fake_logits, clip_embs, device=device)
                d_loss_val = calculate_discriminator_loss(real_scores, fake_scores) + lambda_gp * gp

                discriminator_optimiser.zero_grad()
                d_loss_val.backward()
                discriminator_optimiser.step()

            # train generator
            for _ in range(generator_iterations):
                z = torch.randn(batch_size_current, latent_dim).to(device)
                fake_logits = generator(z, clip_embs)

                fake_probs = torch.softmax(fake_logits / TEMPERATURE, dim=1)
                fake_embed = torch.einsum("bndhw,ne->bedhw", fake_probs, discriminator.texture_embedding.weight)

                # fake = torch.argmax(fake_logits, dim=1).long()

                fake_scores = discriminator(fake_embed, clip_embs, already_embedded=True)

                # entropy regularization (prevents collapsing to too few textures)
                entropy = - (fake_probs * (fake_probs + 1e-12).log()).sum(dim=1).mean()
                delta_entropy = 0.05
                g_loss_val = calculate_generator_loss(fake_scores) - delta_entropy * entropy

                generator_optimiser.zero_grad()
                g_loss_val.backward()
                generator_optimiser.step()

        print(f"Epoch {epoch + 1}/{last_epoch + epochs} | D Loss: {d_loss_val.item():.2f} | G Loss: {g_loss_val.item():.2f} | GP: {gp.item():.2f}")

        # test generator results (to catch mode collapse early)
        probabilities = torch.softmax(fake_logits / TEMPERATURE, dim=1)
        ids, counts = torch.argmax(fake_logits, dim=1).unique(return_counts=True)
        avg_probabilities = probabilities.mean(dim=(0, 2, 3, 4)).detach().cpu().numpy()
        print("\tClasses: ", ids.detach().cpu().numpy())
        print("\tCounts:  ", counts.detach().cpu().numpy())
        print("\tProb:    ", np.round(avg_probabilities, 2))

        if epoch > 0 and epoch % 100 == 0:
            save_model(generator, discriminator, generator_optimiser, discriminator_optimiser, clip_cache, epoch)

    save_model(generator, discriminator, generator_optimiser, discriminator_optimiser, clip_cache, epoch + 1)


def continue_training_gan():
    generator, discriminator, g_opt, d_opt, clip_cache, epoch = load_model()
    generator.train()
    discriminator.train()
    train_gan(generator, discriminator, g_opt, d_opt, clip_cache, epoch)


def save_model(
        generator: Generator3D,
        discriminator: Discriminator3D,
        generator_optimiser: optim.Adam | optim.RMSprop,
        discriminator_optimiser: optim.Adam | optim.RMSprop,
        clip_cache,
        epoch: int):
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'g_optimizer': generator_optimiser.state_dict(),
        'd_optimizer': discriminator_optimiser.state_dict(),
        'clip_cache': clip_cache,
        'epoch': epoch
    }, f"data/model/gan-checkpoint-{epoch}.pth")
    print("Checkpoint saved!")


def load_model(file_path: str = "data/model/gan-checkpoint-50.pth"):
    # load checkpoint
    checkpoint = torch.load(file_path)
    epoch = checkpoint['epoch']
    clip_cache = checkpoint['clip_cache']
    print(f'Loading {MODEL_NAME} (epoch {epoch})')

    # initialize generator
    generator = Generator3D(latent_dim=256)
    generator.load_state_dict(checkpoint['generator'])

    # initialize discriminator
    discriminator = Discriminator3D()
    discriminator.load_state_dict(checkpoint['discriminator'])

    # initialize generator optimiser
    g_opt = optim.Adam(generator.parameters(), lr=2e-5, betas=(0.5, 0.9))
    # g_opt = optim.RMSprop(generator.parameters(), lr=5e-5, alpha=0.9)
    g_opt.load_state_dict(checkpoint['g_optimizer'])

    # initialize discriminator optimiser
    d_opt = optim.Adam(discriminator.parameters(), lr=2e-5, betas=(0.5, 0.9))
    # d_opt = optim.RMSprop(discriminator.parameters(), lr=1e-4, alpha=0.9)
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
