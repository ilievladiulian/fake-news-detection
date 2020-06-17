import torch
import torch.nn.functional as F
import torch.optim as optim
import dataset.load_dataset as load_dataset
from model.cnn_model import CNN
import output_handler
from training_handler import TrainingHandler
import numpy as np

class ConvolutionalNN():
    def __init__(self, embedding, batch_size):
        TEXT, vocab_size, word_embeddings, self.train_iter, self.valid_iter, self.test_iter = load_dataset.load(embedding=embedding, batch_size=batch_size)
        self.embedding = embedding

        batch_size = 4
        output_size = 10
        in_channel = 1
        out_channel = 16
        kernel_heights = [3, 5, 7]
        keep_probab = 1
        stride = 1
        padding = 0
        embedding_length = 300

        self.model = CNN(batch_size, output_size, in_channel, out_channel, kernel_heights, keep_probab, stride, padding, vocab_size, embedding_length, word_embeddings)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), weight_decay=0.0005, lr=0.0001)
        loss_fn = F.cross_entropy
        self.training_handler = TrainingHandler(optimizer, loss_fn)

    def train(self, numberOfEpochs):
        patience_threshold = 3
        patience = patience_threshold
        min_valid_loss = np.Inf
        for epoch in range(numberOfEpochs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            train_loss, train_acc = self.training_handler.train_model(self.model, self.train_iter, epoch)
            val_loss, val_acc = self.training_handler.eval_model(self.model, self.valid_iter)
            print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
            output_handler.outputFileHandler.write(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

            patience -= 1
            if val_loss < min_valid_loss and abs(min_valid_loss - val_loss) > 0.005:
                patience = patience_threshold
                torch.save(self.model, "./saved_models/cnn-" + self.embedding)
                min_valid_loss = val_loss

            if patience == 0:
                break

    def test(self):
        self.model = torch.load("./saved_models/cnn-" + self.embedding)
        test_loss, test_acc = self.training_handler.eval_model(self.model, self.test_iter)
        return test_loss, test_acc