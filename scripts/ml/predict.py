import numpy as np
import torch
from scripts.ml.train_vae import VoxelVAE
from pathlib import Path
from scripts.transformations.randomize_data import get_random_dataset

base_path = Path(__file__).parent
model_file = base_path / '..' / '..' / 'data' / 'model' / 'model.pth'
vocab_file = base_path / '..' / '..' / 'data' / 'model' / 'vocab.pkl'


def predict():
    model = VoxelVAE()
    model.load_state_dict(torch.load(model_file))
    model.to('cpu')
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, model.latent_dim).to('cpu')
        generated = model.decode(z)
        generated_binary = (generated > 0.5).int()
    return generated_binary.cpu().numpy().squeeze()  # shape: (N, 1, 16, 16, 16)