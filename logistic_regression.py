import torch
import torch.nn as nn
import torch.optim as optim
import dataset.load_dataset as load_dataset
from training_handler import TrainingHandler
from model.logistic_regression_model import LogisticRegressionModel

class LogisticRegression():
    def __init__(self):
        datasetType = 'linear'
        TEXT, vocab_size, word_embeddings, self.train_iter, self.valid_iter, self.test_iter = load_dataset.load(datasetType=datasetType)

        output_size = 13
        learning_rate = 2e-5

        self.model = LogisticRegressionModel(vocab_size, output_size)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr = learning_rate)
        self.training_handler = TrainingHandler(optimizer, loss_fn)

    def train(self):
        for epoch in range(10):
            torch.cuda.empty_cache()
            train_loss, train_acc = self.training_handler.train_model(self.model, self.train_iter, epoch)
            val_loss, val_acc = self.training_handler.eval_model(self.model, self.valid_iter)
            print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

    def test(self):
        test_loss, test_acc = self.training_handler.eval_model(self.model, self.test_iter)
        return test_loss, test_acc

