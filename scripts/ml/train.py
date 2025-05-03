import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from scripts.training_data import load_training_data
from scripts.ml.TextTo3D import TextTo3D
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

base_path = Path(__file__).parent
model_file = base_path / '..' / '..' / 'data' / 'model' / 'model.pth'
vocab_file = base_path / '..' / '..' / 'data' / 'model' / 'vocab.pkl'


def build_vocab(texts):
    vocab = defaultdict(lambda: len(vocab))
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    for text in texts:
        for word in text.split():
            _ = vocab[word]
    return vocab


def text_to_tensor(text, vocab):
    indices = [vocab.get(word, vocab["<UNK>"]) for word in text.split()]
    return torch.tensor(indices, dtype=torch.long)


def train_ml_model():
    # load training data
    input_text, output = load_training_data()

    # preprocess text
    vocab = build_vocab(input_text)
    # with open(vocab_file, "wb") as file:
    #    pickle.dump(vocab, file)  # save vocab
    tokenized = [text_to_tensor(t, vocab) for t in input_text]
    padded = pad_sequence(tokenized, batch_first=True, padding_value=0)

    # Convert output to torch
    output_tensor_data = torch.stack([torch.tensor(arr, dtype=torch.float32) for arr in output])

    dataset = TensorDataset(padded, output_tensor_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = TextTo3D(vocab_size=len(vocab))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # todo: make 1e-3

    # training
    for epoch in range(5000):  # adjust as needed
        model.train()
        total_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # save model
    torch.save(model.state_dict(), model_file)
    print('Model saved.')
