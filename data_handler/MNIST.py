from __future__ import division
from tensorflow.examples.tutorials.mnist import input_data
from data_handler.BaseDataset import BaseDataset, DatasetCollection
import numpy as np

Xs = 'Xs'
Xs_img = 'Xs_img'
labels = 'labels'
TRAIN_SIZE = 60000
TEST_SIZE = 10000
LABEL_SIZE = 10


class MNIST_train(BaseDataset):
    SIZE = TRAIN_SIZE
    LABEL_SIZE = LABEL_SIZE
    Xs = Xs
    Xs_img = Xs_img
    labels = labels
    BATCH_KEYS = [
        Xs,
        labels,
        Xs_img
    ]

    def load(self, path, limit=None):
        mnist = input_data.read_data_sets(path, one_hot=True)
        self.data[Xs], self.data[labels] = mnist.train.next_batch(self.SIZE)

    def save(self):
        pass

    def preprocess(self):
        data = self.data[Xs]
        shape = data.shape
        data = np.reshape(data, [shape[0], 28, 28, 1])
        self.data[Xs] = data


class MNIST_test(BaseDataset):
    SIZE = TEST_SIZE
    LABEL_SIZE = LABEL_SIZE
    Xs = Xs
    Xs_img = Xs_img
    labels = labels
    BATCH_KEYS = [
        Xs,
        labels,
        Xs_img
    ]

    def load(self, path, limit=None):
        mnist = input_data.read_data_sets(path, one_hot=True)

        self.data[Xs], self.data[labels] = mnist.test.next_batch(self.SIZE)

    def save(self):
        pass

    def preprocess(self):
        data = self.data[Xs]
        shape = data.shape
        data = np.reshape(data, [shape[0], 28, 28, 1])
        self.data[Xs] = data


class MNIST(DatasetCollection):
    def __init__(self):
        super().__init__()
        self.train_set = MNIST_train()
        self.test_set = MNIST_test()

    def load(self, path, **kwargs):
        super().load(path, **kwargs)
        self.train_set.shuffle()
        self.test_set.shuffle()
