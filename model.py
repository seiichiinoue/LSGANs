import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt


class Generator(nn.Module):
    """
    This model is copies of Least Squares Generative Adversarial Networks.
    """

    def __init__(self):
        
        super(Generator, self).__init__()

        # fully connected layer
        self.fc1 = nn.Linear(in_features=256, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=7 * 7 * 128)
        self.fc2_bn = nn.BatchNorm2d(7 * 7 * 128)
        
        # deconvolutional layer
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=5, stride=2)
        self.deconv1_bn = nn.BatchNorm2d(128)
        
        # output
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=5, stride=2)


    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2_bn(self.fc2(x))
        x = x.view(-1, 128, 7, 7)
        x = self.deconv1_bn(self.deconv1(x))

        return self.deconv2(x)


class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        # deconvolutional layer 
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=320, kernel_size=5, stride=2)
        self.conv2_bn = nn.BatchNorm2d(320)

        # fully connected layer
        self.fc1 = nn.Linear(in_features=128 * 7 * 7, out_features=1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1)


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2_bn(self.deconv2(x))
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc1_bn(self.fc1(x))

        return self.fc2(x)


if __name__ == '__main__':
    print(Generator())
    print(Discriminator())
