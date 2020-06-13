import torch.nn as nn
from torch.nn import functional as F

class LogisticRegressionModel(nn.Module):
    def __init__(self, output_dim, vocab_size, embedding_length, weights):
        super(LogisticRegressionModel, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the word embedding.
        self.linear = nn.Linear(embedding_length, output_dim)

    def forward(self, x):
        input = self.word_embeddings(x)
        out = F.sigmoid(self.linear(input))
        return out