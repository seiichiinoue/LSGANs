import os
import torch
from model import *

def generate(epoch, G, log_dir='data/generated'):
    
    G.eval()
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # generate randoms
    sample_z = torch.rand((128, 256))
    sample_z = Variable(sample_z, volatile=True)
    
    # generate sample via generator
    samples = G(sample_z).data.cpu()
    save_image(samples, os.path.join(log_dir, 'epoch_%03d.png' % (epoch)))


if __name__ == '__main__':

    G = Generator()
    # generate
    generate(epoch=10, G=G)
    