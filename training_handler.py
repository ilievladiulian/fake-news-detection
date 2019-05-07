import torch
from torch.autograd import Variable

class TrainingHandler():
    def __init__(self, optimizer, loss_fn):
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def clip_gradient(self, model, clip_value):
        params = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)
        
    def train_model(self, model, train_iter, epoch):
        total_epoch_loss = 0
        total_epoch_acc = 0
        model.cuda()
        steps = 0
        model.train()
        for idx, batch in enumerate(train_iter):
            text = batch.content[0]
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            if (text.size()[0] is not 4):
                continue
            self.optimizer.zero_grad()
            prediction = model(text)
            loss = self.loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
            acc = 100.0 * num_corrects/len(batch)
            loss.backward()
            self.clip_gradient(model, 1e-1)
            self.optimizer.step()
            steps += 1
            
            if steps % 100 == 0:
                print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
            
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()
            
        return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

    def eval_model(self, model, val_iter):
        total_epoch_loss = 0
        total_epoch_acc = 0
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(val_iter):
                text = batch.content[0]
                if (text.size()[0] is not 4):
                    continue
                target = batch.label
                target = torch.autograd.Variable(target).long()
                if torch.cuda.is_available():
                    text = text.cuda()
                    target = target.cuda()
                prediction = model(text)
                loss = self.loss_fn(prediction, target)
                num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
                acc = 100.0 * num_corrects/len(batch)
                total_epoch_loss += loss.item()
                total_epoch_acc += acc.item()

        return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)