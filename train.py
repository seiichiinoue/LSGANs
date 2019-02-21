import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
# my modules
from model import *


def train(D, G, train_itr, epoch, batch_size=128, z_dim=62):
    
    # settings
    D_optimizer = optim.Adam(D.parameters())
    G_optimizer = optim.Adam(G.parameters())
    
    # labels
    a = 0
    b = 1
    c = 1

    for i in range(epoch):

        # answer labels
        y_real = torch.ones(batch_size, 1)
        y_fake = torch.zeros(batch_size, 1)

        D_running_loss = 0
        G_running_loss = 0

        for batch_index, (real_img, _) in enumerate(train_itr):

            if real_img.size()[0] != batch_size:
                break

            # random generating img seeds
            z = torch.rand((batch_size, z_dim))

            # update discriminator
            D_optimizer.zero_grad()

            # real
            D_real = D(real_img)
            D_real_loss = torch.sum((D_real - b) ** 2)

            # fake
            fake_img = G(z)
            D_fake = D(fake_img.detach())  # stop propagate to G
            D_fake_loss = torch.sum((D_fake - a) ** 2)

            # minimizing loss
            D_loss = 0.5 * (D_real_loss + D_fake_loss) / batch_size
            D_loss.backward()
            D_optimizer.step()
            D_running_loss += D_loss.data.item()

            # update generator
            G_optimizer.zero_grad()

            fake_img = G(z)
            D_fake = D(fake_img)
            
            G_loss = 0.5 * (torch.sum((D_fake - c) ** 2)) / batch_size
            G_loss.backward()
            G_optimizer.step()
            G_running_loss += G_loss.data.item()

        print('epoch: {}, discriminator loss: {}, generator loss: {}'.format(i, D_running_loss, G_running_loss))

    torch.save(G.state_dict(), 'model/generator.pth')
    torch.save(D.state_dict(), 'model/discriminator.pth')


if __name__ == '__main__':
    
    # args
    epoch = 30
    z_dim = 62
    batch_size = 128

    # dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
    train_itr = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # build modules
    D = Discriminator()
    G = Generator()

    train(D, G, train_itr, epoch, batch_size, z_dim)

