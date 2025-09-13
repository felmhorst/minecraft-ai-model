import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm
import torch.optim as optim
import clip
from scripts.training_data.get_random_training_dataset import get_random_training_dataset
from scripts.training_data.normalize_block_ids import get_max_block_id
from scripts.visualize.visualize_voxel_grid import visualize_voxel_grid

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

        input_dimensions = latent_dim + LABEL_EMBED_DIMENSIONS

        self.latent_to_tensor = nn.Sequential(
            nn.Linear(input_dimensions, 128 * 2 * 2 * 2),
            nn.ReLU(True)
        )

        self.deconv1 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.cbn1 = ConditionalBatchNorm3d(64, LABEL_EMBED_DIMENSIONS)

        self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.cbn2 = ConditionalBatchNorm3d(32, LABEL_EMBED_DIMENSIONS)

        self.deconv3 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1)
        self.cbn3 = ConditionalBatchNorm3d(16, LABEL_EMBED_DIMENSIONS)

        # +3 for positional encoding
        channels_with_positional_encoding = 16 + 3

        # occupancy head to determine whether a block is solid
        self.to_occupancy = nn.Conv3d(channels_with_positional_encoding, 1, kernel_size=3, padding=1)

        # texture head to determine the block (other than air)
        self.to_texture = nn.Conv3d(channels_with_positional_encoding, NUM_TEXTURES, kernel_size=3, padding=1)

    def forward(
            self,
            z: torch.Tensor,
            label_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        x = torch.relu(self.cbn1(self.deconv1(x), label_embeddings))  # [batch_size, 64, 4, 4, 4]
        x = torch.relu(self.cbn2(self.deconv2(x), label_embeddings))  # [batch_size, 32, 8, 8, 8]
        x = torch.relu(self.cbn3(self.deconv3(x), label_embeddings))  # [batch_size, 16, 16, 16, 16]

        # add positional encoding
        B, _, D, H, W = x.shape
        pos = self.get_positional_grid(D, x.device).unsqueeze(0).repeat(B, 1, 1, 1, 1)
        x = torch.cat([x, pos], dim=1)  # [batch_size, 19, 16, 16, 16] (added 3 channels for positional encoding)

        occupancy_logits = self.to_occupancy(x)  # [batch_size, 1, 16, 16, 16]
        texture_logits = self.to_texture(x)      # [batch_size, NUM_TEXTURES, 16, 16, 16]

        return occupancy_logits, texture_logits

    def sample_texture_ids(
            self,
            z: torch.Tensor,
            label_embeddings: torch.Tensor) -> torch.Tensor:
        """Samples outputs based on a (randomized) input tensor z and text prompt."""
        occupancy_logits, texture_logits = self.forward(z, label_embeddings)

        occupancy = (torch.sigmoid(occupancy_logits) > 0.5).long()   # [batch_size, 1, 16, 16, 16]
        texture = torch.argmax(texture_logits, dim=1, keepdim=True)  # [batch_size, 1, 16, 16, 16]

        voxel_ids = texture * occupancy  # [batch_size, 16, 16, 16]
        return voxel_ids

    @staticmethod
    def get_positional_grid(
            size: int,
            device: str | torch.device) -> torch.Tensor:
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

        in_channels = TEXTURE_EMBED_DIMENSIONS + 1  # texture embedding + occupancy

        self.conv_layers = nn.Sequential(
            # nn.Conv3d(in_channels, 16, kernel_size=4, stride=2, padding=1),  # [batch_size, 16, 8, 8, 8]
            spectral_norm(nn.Conv3d(in_channels, 16, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm3d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),  # [batch_size, 32, 4, 4, 4]
            spectral_norm(nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm3d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),  # [batch_size, 64, 2, 2, 2]
            spectral_norm(nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm3d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # self.fc = nn.Linear(64 * 2 * 2 * 2, 1)  # [batch_size, 1]
        self.fc = spectral_norm(nn.Linear(64 * 2 * 2 * 2, 1))

        self.embed_proj = nn.Linear(LABEL_EMBED_DIMENSIONS, 64 * 2 * 2 * 2)  # [batch_size, 512]

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
        x = torch.cat([occupancy.float(), texture_logits], dim=1)
        batch_size = x.size(0)
        features = self.conv_layers(x).reshape(batch_size, -1)  # [batch_size, 512]

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
    real_texture_embed = discriminator.texture_embedding(real_textures).permute(0, 4, 1, 2, 3)  # [batch_size, TEXTURE_EMBED_DIMENSIONS, 16, 16, 16]

    # embed textures (generated batch)
    fake_texture_probs = torch.softmax(fake_texture_logits / TEMPERATURE, dim=1)  # [batch_size, NUM_TEXTURES, 16, 16, 16] (ratios)
    fake_texture_embed = torch.einsum("bndhw,ne->bedhw", fake_texture_probs, discriminator.texture_embedding.weight)  # [batch_size, TEXTURE_EMBED_DIMENSIONS, 16, 16, 16]

    fake_occupancy_probs = torch.sigmoid(fake_occupancy_logits).float()  # [batch_size, 1, 16, 16, 16]
    real_occupancy_float = real_occupancy.float()  # [batch_size, 1, 16, 16, 16]

    # combine occupancy + texture
    real_x = torch.cat([real_occupancy_float, real_texture_embed], dim=1)
    fake_x = torch.cat([fake_occupancy_probs, fake_texture_embed], dim=1)

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

    # compute per-voxel squared norms
    gradients_sq = gradients.pow(2)  # [batch_size, 1+TEXTURE_EMBED_DIM, 16, 16, 16]

    # compute squared occupancy gradient
    occupancy_per_voxel_sq = gradients_sq[:, 0, ...]  # [batch_size, 16, 16, 16]
    occupancy_gradient_sq = occupancy_per_voxel_sq.sum(dim=[1, 2, 3])  # [batch_size]

    # computed weighted squared texture gradient
    # the texture's gradient depends on the occupancy. an occupancy closer to 1 is more heavily penalized
    texture_per_voxel_sq = gradients_sq[:, 1:, ...].sum(dim=1)  # [batch_size, 16, 16, 16]
    occupancy_interpolated = interpolated[:, 0, ...]  # [batch_size, 16, 16, 16]
    weighted_texture_sq = texture_per_voxel_sq * occupancy_interpolated  # [batch_size, 16, 16, 16]
    texture_gradient_sq = weighted_texture_sq.sum(dim=[1, 2, 3])  # [batch_size]

    # combine gradients + compute penalty
    gradient_norm = torch.sqrt(occupancy_gradient_sq + texture_gradient_sq + 1e-12)  # todo
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()  # [] (single scalar)
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
        alpha_texture: float = 0.0):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if generator is None:
        generator = Generator3D(latent_dim=latent_dim)
    if discriminator is None:
        discriminator = Discriminator3D()
    if generator_optimiser is None:
        # generator_optimiser = optim.RMSprop(generator.parameters(), lr=1e-4, alpha=0.9)
        generator_optimiser = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    if discriminator_optimiser is None:
        # discriminator_optimiser = optim.RMSprop(discriminator.parameters(), lr=1e-4, alpha=0.9)
        discriminator_optimiser = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

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
            real_occupancy = real_occupancy.to(device).long()
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
                occupancy_loss = nn.functional.binary_cross_entropy_with_logits(occupancy_logits, real_occupancy.float())

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

        # test generator results (to catch mode collapse early)
        # probabilities = torch.softmax(texture_logits / TEMPERATURE, dim=1)
        # ids, counts = torch.argmax(texture_logits, dim=1).unique(return_counts=True)
        # avg_probabilities = probabilities.mean(dim=(0, 2, 3, 4)).detach().cpu().numpy()
        # print("\tClasses: ", ids.detach().cpu().numpy())
        # print("\tCounts:  ", counts.detach().cpu().numpy())
        # print("\tProb:    ", np.round(avg_probabilities, 2))

        if epoch > 0 and epoch % 10 == 0:
            clip_embedding = get_clip_embedding("solid cuboid")
            test_model(generator, clip_embedding, latent_dim, device)
            clip_embedding = get_clip_embedding("solid pyramid")
            test_model(generator, clip_embedding, latent_dim, device)
            clip_embedding = get_clip_embedding("solid sphere")
            test_model(generator, clip_embedding, latent_dim, device)
            clip_embedding = get_clip_embedding("gable house")
            test_model(generator, clip_embedding, latent_dim, device)
        if epoch > 0 and epoch % 100 == 0:
            save_model(generator, discriminator, generator_optimiser, discriminator_optimiser, clip_cache, epoch)

    save_model(generator, discriminator, generator_optimiser, discriminator_optimiser, clip_cache, epoch + 1)


def test_model(generator: Generator3D, clip_embedding: torch.Tensor, latent_dim: int, device: torch.device | str):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(1, latent_dim, device=device)
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
    visualize_voxel_grid(data_np)
    generator.train()


def continue_training_gan(
        last_epoch: int = 100,
        epochs: int = 100,
        lr: float = 1e-4,
        alpha_adversarial: float = 1.0,
        alpha_occupancy: float = 0.0,
        alpha_texture: float = 0.0
):
    generator, discriminator, g_opt, d_opt, clip_cache, epoch = load_model(file_path=f"data/model/gan-checkpoint-{last_epoch}.pth")
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


def load_model(file_path: str = "data/model/gan-checkpoint-300.pth"):
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
    g_opt = optim.Adam(generator.parameters(), lr=2e-6, betas=(0.5, 0.9))
    # g_opt = optim.RMSprop(generator.parameters(), lr=5e-5, alpha=0.9)
    g_opt.load_state_dict(checkpoint['g_optimizer'])

    # initialize discriminator optimiser
    d_opt = optim.Adam(discriminator.parameters(), lr=2e-6, betas=(0.5, 0.9))
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


def train_gan_by_schedule():
    train_gan(epochs=100, lr=2e-4, alpha_adversarial=0.5, alpha_occupancy=3.0, alpha_texture=1.0)
    continue_training_gan(last_epoch=100, epochs=100, lr=2e-4, alpha_adversarial=0.5, alpha_occupancy=1.0, alpha_texture=3.0)
    continue_training_gan(last_epoch=200, epochs=100, lr=1e-4, alpha_adversarial=1.0, alpha_occupancy=0.5, alpha_texture=0.5)
    # continue_training_gan(last_epoch=50, epochs=50, lr=1e-4, alpha_occupancy=2.0, alpha_texture=2.0)
    # continue_training_gan(last_epoch=100, epochs=50, lr=5e-5, alpha_occupancy=1.0, alpha_texture=1.0)
    # continue_training_gan(last_epoch=150, epochs=50, lr=5e-5, alpha_occupancy=0.2, alpha_texture=0.2)
    # continue_training_gan(last_epoch=200, epochs=100, lr=2e-5, alpha_occupancy=0.0, alpha_texture=0.0)
    # continue_training_gan(last_epoch=300, epochs=100, lr=1e-5, alpha_occupancy=0.0, alpha_texture=0.0)
    # continue_training_gan(last_epoch=400, epochs=100, lr=1e-6, alpha_occupancy=0.0, alpha_texture=0.0)
