import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard

class AUTOMAP(nn.Module):

    def __init__(self, n):
        self.fc2 = nn.Linear(in_features = 2*n**2, out_features=n**2)
        self.fc3 = nn.Linear(in_features = n**2, out_features=n**2)

        self.c1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=5, stride=1)
        self.c2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=5, stride=1)

        self.c3 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=7, stride=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        n = x.shape[1]
        #convert to single column vector
        real_part = x.real.view(-1)
        imag_part = x.imag.view(-1)
        x = torch.cat((real_part, imag_part), dim=0)

        x = self.fc2(x)
        x = self.tanh(x)

        x = self.fc3(x)
        x = self.tanh(x)

        x = x.reshape((-1 , n, n))

        x = self.c1(x)
        x = self.relu(x)

        x = self.c2(x)
        x = self.relu(x)

        x = self.c3(x)

        return x
