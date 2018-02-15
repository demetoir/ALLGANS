from __future__ import division
from tensorflow.examples.tutorials.mnist import input_data
from data_handler.AbstractDataset import AbstractDataset
from env_settting import MNIST_PATH
from dict_keys.dataset_batch_keys import *
from dict_keys.input_shape_keys import *
import numpy as np


class MNIST(AbstractDataset):
    TRAIN_SIZE = 60000
    TEST_SIZE = 10000
    LABEL_SIZE = 10

    def __init__(self, preprocess=None, batch_after_task=None):
        super().__init__(preprocess, batch_after_task)
        self.batch_keys = [BATCH_KEY_TRAIN_X, BATCH_KEY_TRAIN_LABEL, BATCH_KEY_TEST_X, BATCH_KEY_TEST_LABEL]

        def dummy():
            pass

        self.before_load_task = dummy

    def __repr__(self):
        return self.__class__.__name__

    def load(self, path, limit=None):
        mnist = input_data.read_data_sets(path, one_hot=True)

        if limit is None:
            self.__train_size = self.TRAIN_SIZE
            self.__test_size = self.TEST_SIZE
        else:
            self.__train_size = limit
            self.__test_size = limit
        train_x, train_label = mnist.train.next_batch(self.__train_size)
        test_x, test_label = mnist.test.next_batch(self.__test_size)

        self.data = {
            BATCH_KEY_TRAIN_X: train_x,
            BATCH_KEY_TRAIN_LABEL: train_label,
            BATCH_KEY_TEST_X: test_x,
            BATCH_KEY_TEST_LABEL: test_label
        }

    def save(self):
        raise NotImplementedError


class MNISTHelper:
    @staticmethod
    def preprocess(dataset):
        # original MNIST image size is 28*28 but need to resize 32*32
        data = dataset.data[BATCH_KEY_TRAIN_X]
        shape = data.shape
        data = np.reshape(data, [shape[0], 28, 28])
        npad = ((0, 0), (2, 2), (2, 2))
        data = np.pad(data, pad_width=npad, mode='constant', constant_values=0)
        data = np.reshape(data, [shape[0], 32, 32, 1])
        dataset.data[BATCH_KEY_TRAIN_X] = data

        data = dataset.data[BATCH_KEY_TEST_X]
        shape = data.shape
        data = np.reshape(data, [shape[0], 28, 28])
        npad = ((0, 0), (2, 2), (2, 2))
        data = np.pad(data, pad_width=npad, mode='constant', constant_values=0)
        data = np.reshape(data, [shape[0], 32, 32, 1])
        dataset.data[BATCH_KEY_TEST_X] = data

    @staticmethod
    def next_batch_task(batch):
        x, label = batch[0], batch[1]
        return x, label

    @staticmethod
    def load_dataset(limit=None):
        mnist_data = MNIST(preprocess=MNISTHelper.preprocess)
        mnist_data.load(MNIST_PATH, limit=limit)
        input_shapes = {
            INPUT_SHAPE_KEY_DATA_X: [32, 32, 1],
            INPUT_SHAPE_KEY_LABEL: [10],
            INPUT_SHAPE_KEY_LABEL_SIZE: 10,
        }

        return mnist_data, input_shapes
