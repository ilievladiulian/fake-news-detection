import dataset.load_dataset as load_dataset
import torch
import torch.nn.functional as F
import torch.optim as optim
from model.rcnn import RCNN
from training_handler import TrainingHandler


TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_dataset.load()

learning_rate = 2e-5
batch_size = 4
output_size = 13
hidden_size = 256
embedding_length = 300

model = RCNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
loss_fn = F.cross_entropy
training_handler = TrainingHandler(optimizer, loss_fn)

for epoch in range(10):
    torch.cuda.empty_cache()
    train_loss, train_acc = training_handler.train_model(model, train_iter, epoch)
    val_loss, val_acc = training_handler.eval_model(model, valid_iter)

    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
    
test_loss, test_acc = training_handler.eval_model(model, test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')