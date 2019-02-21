import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torchvision.utils import save_image


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        # fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 7 * 7 * 128),
            nn.BatchNorm1d(7 * 7 * 128),
            nn.ReLU()
        )
        # deconvolutional layer
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=5, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 7, 7)
        x = self.deconv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # convolutional layer
        self.conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(256, 320, kernel_size=5, stride=2),
            nn.BatchNorm2d(320),
            nn.LeakyReLU()
        )    
        # fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # check model shape
    print(Generator())
    print(Discriminator())
