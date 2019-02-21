import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from model import *

def train_itr(dataset, batch_size=128):
    
    idx = np.random.permutation(len(dataset.train_labels))
    X, y = [], []
    for j, i in enumerate(idx):
        X.append(dataset.train_data[i])
        y.append(dataset.train_labels[i])
        if j + 1 % batch_size == 0:
            yield (torch.from_numpy(np.array(X)).float(), torch.from_numpy(np.array(y)).float())
            X, y = [], []


def train(dataset, epoch, batch_size=128, z_dim=28 * 28):

    D = Discriminator()
    G = Generator()

    criterion = nn.MSELoss()
    D_optimizer = optim.Adam(D.parameters())
    G_optimizer = optim.Adam(G.parameters())

    for i in range(epoch):

        # answer labels
        y_real = torch.ones(batch_size, 1)
        y_fake = torch.zeros(batch_size, 1)

        D_running_loss = 0
        G_running_loss = 0

        for batch_index, (real_img, _) in enumerate(train_itr(dataset=dataset, batch_size=batch_size)):

            if real_img.size()[0] != batch_size:
                break

            # random generating img seeds
            z = torch.rand((batch_size, z_dim))

            # update discriminator
            D_optimizer.zero_grad()

            # real
            D_real = D(real_img)
            D_real_loss = criterion(D_real, y_real)

            # fake
            faek_img = G(z)
            D_fake = D(faek_img.detach())  # stop propagate to G
            D_fake_loss = criterion(D_fake, y_fake)

            # minimizing loss
            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            D_optimizer.step()
            D_running_loss += D_loss.data.item()

            # update generator
            G_optimizer.zero_grad()

            fake_img = G(z)
            D_fake = D(fake_img)
            
            G_loss = criterion(D_fake, y_real)
            G_loss.backward()
            G_optimizer.step()
            G_running_loss += G_loss.data.item()

            print('epoch: {}, discriminator loss: {}, generator loss: {}'.format(i, D_running_loss, G_running_loss))

    torch.save(G.state_dict(), 'model/generator.pth')
    torch.save(D.state_dict(), 'model/discriminator.pth')

if __name__ == '__main__':
    
    dataset = datasets.MNIST('data/mnist', train=True, download=True)  
    epoch = 30
    z_dim = dataset.train_data[0].shape[0] ** 2
    batch_size = 128

    train(dataset, epoch, batch_size, z_dim)


