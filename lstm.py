import torch
import torch.nn.functional as F
import torch.optim as optim
import dataset.load_dataset as load_dataset
from model.lstm_model import LSTMClassifier
import output_handler
from training_handler import TrainingHandler

class LongShortTermMemory():
    def __init__(self):
        datasetType = 'generic'
        TEXT, vocab_size, word_embeddings, self.train_iter, self.valid_iter, self.test_iter = load_dataset.load(datasetType=datasetType)

        batch_size = 4
        output_size = 13
        hidden_size = 256
        embedding_length = 300

        self.model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), weight_decay=0.0005, lr=0.0001)
        loss_fn = F.cross_entropy
        self.training_handler = TrainingHandler(optimizer, loss_fn)

    def train(self, numberOfEpochs):
        for epoch in range(numberOfEpochs):
            torch.cuda.empty_cache()
            train_loss, train_acc = self.training_handler.train_model(self.model, self.train_iter, epoch)
            val_loss, val_acc = self.training_handler.eval_model(self.model, self.valid_iter)
            print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
            output_handler.outputFileHandler.write(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

    def test(self):
        test_loss, test_acc = self.training_handler.eval_model(self.model, self.test_iter)
        return test_loss, test_acc