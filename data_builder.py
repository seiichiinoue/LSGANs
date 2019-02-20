import gzip, os, six, sys
from six.moves.urllib import request
from PIL import Image
import numpy as np

parent = 'http://yann.lecun.com/exdb/mnist'
train_images_filename = 'train-images-idx3-ubyte.gz'
train_labels_filename = 'train-labels-idx1-ubyte.gz'
test_images_filename = 't10k-images-idx3-ubyte.gz'
test_labels_filename = 't10k-labels-idx1-ubyte.gz'

n_train = 60000
n_test = 10000
dim = 28 * 28

def load_mnist(data_filename, label_filename, num):

    images = np.zeros(num * dim)
    label = np.zeros(num).reshape((num, ))

    with gzip.open(data_filename, mode='rb') as f_images, gzip.open(label_filename, mode='rb') as f_labels:
        f_images.read(16)
        f_labels.read(8)

        for i in range(num):
            label[i] = f_labels.read(1)
            for j in range(dim):
                images[i, j] = f_images.read(1)

            if i % 100 == 0 or i == num - 1:
                sys.stdout.write('\rloading images... ({}/{})'.format(i, num))
                sys.stdout.flush()

    sys.stdout.write('\n')

    images = (images / 255.0 * 2.0) - 1.0
    return images, label


def download_mnist_data():
    request.urlretrieve('{}/{}'.format(parent, train_images_filename), train_images_filename)
    request.urlretrieve('{}/{}'.format(parent, train_labels_filename), train_labels_filename)
    request.urlretrieve('{}/{}'.format(parent, test_images_filename), test_images_filename)
    request.urlretrieve('{}/{}'.format(parent, test_labels_filename), test_labels_filename)


def load_train_images():
    images, labels = load_mnist(train_images_filename, train_labels_filename, n_train)
    return images, labels


def load_test_images():
    images, labels = load_mnist(test_images_filename, test_labels_filename, n_test)
    return images, labels
    