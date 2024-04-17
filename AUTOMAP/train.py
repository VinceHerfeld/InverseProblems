import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from datasets import ImageDataset
from models import AUTOMAP
import time

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)

def print_summary(epoch, i, nb_batch, loss, batch_time,
                  average_loss, average_time, mode):
    '''
        mode = Train or Test
    '''
    summary = '[' + str(mode) + '] Epoch: [{0}][{1}/{2}]\t'.format(
        epoch, i, nb_batch)

    string = ''
    string += ('Dice Loss {:.4f} ').format(loss)
    string += ('(Average {:.4f}) \t').format(average_loss)
    string += ('Batch Time {:.4f} ').format(batch_time)
    string += ('(Average {:.4f}) \t').format(average_time)

    summary += string
    print(summary)

def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']
    val_loss = state['val_loss']
    filename = save_path + '/' + \
        'model.{:02d}--{:.3f}.pth.tar'.format(epoch, val_loss)
    torch.save(state, filename)

def train(model, dataloader, criterion, optimizer, n_epochs = 10, batch_size = 64, mod = 10):
    model.train()
    for epoch in range(n_epochs):
        epoch_time_sum = []
        epoch_loss_sum = []
        for i, (input_batch, output_batch) in enumerate(dataloader, 1):
            start = time.time()

            input_batch, output_batch = input_batch.to(device), output_batch.to(device)
            y = model(input_batch)

            optimizer.zero_grad()
            loss = criterion(y, output_batch)
            loss.backward()
            optimizer.step()

            batch_time = time.time() - start

            epoch_time_sum += [batch_time]
            epoch_loss_sum += [loss.item()]

            average_time = torch.mean(epoch_time_sum)
            average_loss = torch.mean(epoch_loss_sum)

            if i % mod == 0:
                print_summary(epoch + 1, i, len(dataloader), loss, batch_time, average_loss, average_time)



Dataset = ImageDataset("./trainset")
Loader = DataLoader(Dataset, batch_size=1, shuffle=True)

n = Dataset[0][0].shape[1]

model = AUTOMAP(n = n)
model.to(device)

learning_rate = 1e-3

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train(model, Loader, criterion, optimizer, n_epochs = 10, batch_size = 64, mod = 10)




