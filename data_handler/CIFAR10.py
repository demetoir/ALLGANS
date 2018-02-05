from data_handler.AbstractDataset import AbstractDataset
from dict_keys.dataset_batch_keys import *
from glob import glob
import os
import pickle


class CIFAR10(AbstractDataset):
    _PATTERN_TRAIN_FILE = "data_batch_*"
    _PATTERN_TEST_FILE = "test_batch"
    _PKCL_KEY_TRAIN_DATA = b"data"
    _PKCL_KEY_TRAIN_LABEL = b"labels"
    _PKCL_KEY_TEST_DATA = b"data"
    _PKCL_KEY_TEST_LABEL = b"labels"
    LABEL_SIZE = 10

    def __init__(self, preprocess=None, batch_after_task=None):
        super().__init__(preprocess, batch_after_task)
        self.batch_keys = [BATCH_KEY_TRAIN_X, BATCH_KEY_TRAIN_LABEL, BATCH_KEY_TEST_X, BATCH_KEY_TEST_LABEL]
        self._SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self._SOURCE_FILE = 'cifar-10-python.tar.gz'
        self._data_files = [
            "data_batch_1",
            "data_batch_2",
            "data_batch_3",
            "data_batch_4",
            "data_batch_5",
            "test_batch",
            "batches.meta"
        ]

    def load(self, path, limit=None):
        # load train data
        files = glob(os.path.join(path, self._PATTERN_TRAIN_FILE))
        files.sort()
        for file in files:
            with open(file, 'rb') as fo:
                dict_ = pickle.load(fo, encoding='bytes')

            x = dict_[self._PKCL_KEY_TRAIN_DATA]
            self._append_data(BATCH_KEY_TRAIN_X, x)

            label = dict_[self._PKCL_KEY_TRAIN_LABEL]
            self._append_data(BATCH_KEY_TRAIN_LABEL, label)

        # load test data
        files = glob(os.path.join(path, self._PATTERN_TEST_FILE))
        files.sort()
        for file in files:
            with open(file, 'rb') as fo:
                dict_ = pickle.load(fo, encoding='bytes')

            x = dict_[self._PKCL_KEY_TEST_DATA]
            self._append_data(BATCH_KEY_TEST_X, x)

            label = dict_[self._PKCL_KEY_TEST_LABEL]
            self._append_data(BATCH_KEY_TEST_LABEL, label)

    def save(self):
        raise NotImplementedError
