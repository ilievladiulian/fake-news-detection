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
        input = input.permute(0, 2, 1) # y.size() = (batch_size, hidden_size, num_sequences)
        input = F.max_pool1d(input, input.size()[2]) # y.size() = (batch_size, hidden_size, 1)
        input = input.squeeze(2)
        # input = input.permute(1, 0, 2)
        out = F.sigmoid(self.linear(input))
        return out