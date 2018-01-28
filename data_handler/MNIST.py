from __future__ import division
from tensorflow.examples.tutorials.mnist import input_data
from data_handler.AbstractDataset import AbstractDataset
from dict_keys.dataset_batch_keys import *


class MNIST(AbstractDataset):
    TRAIN_SIZE = 60000
    TEST_SIZE = 10000
    LABEL_SIZE = 10

    def __init__(self, preprocess=None, batch_after_task=None):
        super().__init__(preprocess, batch_after_task)
        self.batch_keys = [BATCH_KEY_TRAIN_X, BATCH_KEY_TRAIN_LABEL, BATCH_KEY_TEST_X, BATCH_KEY_TEST_LABEL]

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

        super().load(path, limit)

    def save(self):
        raise NotImplementedError
