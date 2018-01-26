from data_handler.AbstractDataset import AbstractDataset
import pandas as pd
import numpy as np
import os


# TODO implement to label...
class Fashion_MNIST(AbstractDataset):
    PATTERN_TRAIN = "fashion-mnist_train.csv"
    PATTERN_TEST = "fashion-mnist_test.csv"
    TRAIN_x = 'train_x'
    TRAIN_label = 'train_label'
    TEST_x = 'test_x'
    TEST_label = "test_label"
    LABEL_SIZE = 10

    def __init__(self, preprocess=None, batch_after_task=None):
        super().__init__(preprocess, batch_after_task)
        self.batch_keys = [self.TRAIN_x, self.TRAIN_label, self.TEST_x, self.TEST_label]

    def load(self, path, limit=None):
        # TODO load from byte file not csv

        # load train data
        train_data = pd.read_csv(os.path.join(path, self.PATTERN_TRAIN), sep=',', header=None)
        train_data = np.array(train_data)
        train_x = train_data[1:, 1:]
        train_label = train_data[1:, :1]
        self.data[self.TRAIN_x] = train_x
        self.data[self.TRAIN_label] = train_label

        # load test data
        test_data = pd.read_csv(os.path.join(path, self.PATTERN_TEST), sep=',', header=None)
        test_data = np.array(test_data)
        test_x = test_data[1:, 1:]
        test_label = test_data[1:, :1]
        self.data[self.TEST_x] = test_x
        self.data[self.TEST_label] = test_label

        super().load(path, limit)

    def save(self):
        raise NotImplementedError
