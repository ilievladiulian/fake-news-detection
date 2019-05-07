import torch
import torch.nn as nn
import torch.optim as optim
from training_handler import TrainingHandler
from model.linear_regression_model import LinearRegressionModel
import dataset.load_dataset_linear as load_dataset


TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_dataset.load()

output_size = 13
learning_rate = 2e-5

model = LinearRegressionModel(vocab_size, output_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
training_handler = TrainingHandler(optimizer, loss_fn)

for epoch in range(10):
    torch.cuda.empty_cache()
    train_loss, train_acc = training_handler.train_model(model, train_iter, epoch)
    val_loss, val_acc = training_handler.eval_model(model, valid_iter)
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
    
test_loss, test_acc = training_handler.eval_model(model, test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

