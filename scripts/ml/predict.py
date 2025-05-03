import torch

from scripts.training_data import load_training_data
from scripts.ml.TextTo3D import TextTo3D
from scripts.ml.train import text_to_tensor, build_vocab
from pathlib import Path

base_path = Path(__file__).parent
model_file = base_path / '..' / '..' / 'data' / 'model' / 'model.pth'
vocab_file = base_path / '..' / '..' / 'data' / 'model' / 'vocab.pkl'


def predict(text):
    input_text, output = load_training_data()
    vocab = build_vocab(input_text)
    # with open(vocab_file, "rb") as f:
    #     vocab = pickle.load(f)
    model = TextTo3D(vocab_size=len(vocab))  # make sure to re-init the model
    model.load_state_dict(torch.load(model_file))
    model.eval()

    x = text_to_tensor(text, vocab).unsqueeze(0)
    with torch.no_grad():
        return model(x)
