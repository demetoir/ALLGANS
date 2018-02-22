from data_handler.AbstractDataset import AbstractDataset, DownloadInfo
from env_settting import CIFAR100_PATH
from dict_keys.dataset_batch_keys import *
import pickle
import os
import numpy as np


class CIFAR100(AbstractDataset):
    _PATTERN_TRAIN_FILE = "train"
    _PATTERN_TEST_FILE = "test"
    _pkcl_key_train_data = b"data"
    _pkcl_key_train_fine_labels = b"fine_labels"
    _pkcl_key_train_coarse_labels = b"coarse_labels"
    _pkcl_key_test_data = b"data"
    _pkcl_key_test_fine_labels = b"fine_labels"
    _pkcl_key_test_coarse_labels = b"coarse_labels"

    def __init__(self, preprocess=None, batch_after_task=None):
        super().__init__(preprocess, batch_after_task)
        self.batch_keys = [
            BATCH_KEY_TRAIN_X,
            BATCH_KEY_TRAIN_COARSE_LABELS,
            BATCH_KEY_TRAIN_FINE_LABELS,
            BATCH_KEY_TEST_X,
            BATCH_KEY_TEST_COARSE_LABELS,
            BATCH_KEY_TEST_FINE_LABELS
        ]

        self.download_infos = [
            DownloadInfo(
                url='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
                is_zipped=True,
                download_file_name="cifar-100-python.tar.gz",
                extracted_file_names=[
                    "meta",
                    "test",
                    "train",
                    "file.txt~"
                ]
            )
        ]

    def load(self, path, limit=None):
        # load train data
        file_path = os.path.join(path, self._PATTERN_TRAIN_FILE)
        with open(file_path, 'rb') as fo:
            dict_ = pickle.load(fo, encoding='bytes')
        x = dict_[self._pkcl_key_train_data]
        coarse_labels = dict_[self._pkcl_key_train_coarse_labels]
        fine_labels = dict_[self._pkcl_key_train_fine_labels]
        self._append_data(BATCH_KEY_TRAIN_X, x)
        self._append_data(BATCH_KEY_TRAIN_COARSE_LABELS, coarse_labels)
        self._append_data(BATCH_KEY_TRAIN_FINE_LABELS, fine_labels)

        # load test data
        file_path = os.path.join(path, self._PATTERN_TEST_FILE)
        with open(file_path, 'rb') as fo:
            dict_ = pickle.load(fo, encoding='bytes')
        x = dict_[self._pkcl_key_test_data]
        coarse_labels = dict_[self._pkcl_key_test_coarse_labels]
        fine_labels = dict_[self._pkcl_key_test_fine_labels]
        self._append_data(BATCH_KEY_TEST_X, x)
        self._append_data(BATCH_KEY_TEST_COARSE_LABELS, coarse_labels)
        self._append_data(BATCH_KEY_TEST_FINE_LABELS, fine_labels)

    def save(self):
        raise NotImplementedError


class CIFAR100Helper:
    @staticmethod
    def preprocess(dataset):
        # convert image format from NCWH to NWHC
        data = dataset.data[BATCH_KEY_TRAIN_X]
        data = np.reshape(data, [-1, 3, 32, 32])
        data = np.transpose(data, [0, 2, 3, 1])
        dataset.data[BATCH_KEY_TRAIN_X] = data

        data = dataset.data[BATCH_KEY_TEST_X]
        data = np.reshape(data, [-1, 3, 32, 32])
        data = np.transpose(data, [0, 2, 3, 1])
        dataset.data[BATCH_KEY_TEST_X] = data

    @staticmethod
    def next_batch_task(batch):
        x = batch[0]
        return x

    @staticmethod
    def load_dataset(limit=None):
        cifar100 = CIFAR100(preprocess=CIFAR100Helper.preprocess)
        cifar100.load(CIFAR100_PATH, limit=limit)
        return cifar100, [32, 32, 3]
