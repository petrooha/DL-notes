# SkipGram MODEL

# n_embed – the size of the dictionary of embeddings, or how many rows you'll want in the embedding weight matrix
# n_vocab – the size of each embedding vector; the embedding dimension

import torch
from torch import nn
import torch.optim as optim

class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()

        self.embed = nn.Embedding(n_vocab, n_embed)
        self.output = nn.Linear(n_embed, n_vocab)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embed(x)
        x = self.output(x)
        x = self.log_softmax(x)

        return x
