# ML Guidelines
After some experimenting with VAEs, GANs and diffusion models, GAN seems to be the most promising approach.

Minecraft Structures are a type of sparse voxel data (because the majority of blocks is the background class - air). The
model is thus vulnerable to mode collapse.

## WGAN
In comparison to a regular GAN, a Wasserstein GAN provides more meaningful feedback gradients. This reduces the
likelihood of mode collapse and vanishing gradients and thus makes the training more stable.

- spectral normalization
- gradient penalty
- **gradient penalty weighting** (`lambda_gp`): tells the optimizer how strongly to enforce the Lipschitz contraint.
  Start with 10. Increasing the value stabilizes training but slows down learning.

### Generator

### Discriminator (Critic)

### Optimiser
The optimiser updates the parameters of the generator or discriminator based on the gradients computed from the loss.
There are two recommended options: RMSProp or Adam.
(see this [paper on GAN-based generation of 3D data](https://www.sciencedirect.com/science/article/pii/S1361841524000252))
- **Adam**: recommended for GAN (with `beta1 = 0.5`).
- **RMSProp**: recommended for WGAN, but did not work well in my experiments.

### Positional Encoding
Positional encoding helps the generator understand the position it is operating at and thus reduce artifacts. Don't
modify.

### Label Embeddings
Label embeddings are necessary to allow generation through text-prompts. The model uses CLIP ViT-B/32.
- **label embedding dimensions** (`LABEL_EMBED_DIMENSIONS`): The number of dimensions to embed the texture. Must be 512
  for CLIP ViT-B/32.

### Texture Embeddings
Texture embeddings are necessary to help the model learn the meaning of block/texture ids. Since these are not linear,
they should not be passed to the model in a single dimension (i.e. ids).
- **texture embedding dimensions** (`TEXTURE_EMBED_DIMENSIONS`): the number of dimensions to embed the texture. Should
  be between `sqrt(NUM_TEXTURES)` and `NUM_TEXTURES`.