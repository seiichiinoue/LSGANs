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
    """This model is copies of Least Squares Generative Adversarial Networks."""

    def __init__(self):
        
        super(Generator, self).__init__()

        # fully connected layer
        self.fc = nn.Linear(in_features=1024, out_features=256)
        
        # deconvolutional layer
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2)
        self.deconv6 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2)
        self.deconv7 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1)


    def forward(self, x):

