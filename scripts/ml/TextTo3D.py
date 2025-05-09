import torch.nn as nn

class TextTo3D(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, output_shape=(16, 16, 16)):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_shape = output_shape
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_shape[0] * output_shape[1] * output_shape[2])

    def forward(self, x):
        embedded = self.embedding(x)                   # (B, T, E)
        _, (hidden, _) = self.lstm(embedded)           # (1, B, H)
        hidden = hidden.squeeze(0)                     # (B, H)
        out = self.fc(hidden)                          # (B, 4096)
        return out.view(-1, *self.output_shape)        # (B, 16, 16, 16)

    def reset_vocab_size(self, new_vocab_size):
        """Reinitialize embedding layer with new vocab size."""
        self.vocab_size = new_vocab_size
        self.embedding = nn.Embedding(new_vocab_size, self.embedding_dim)
