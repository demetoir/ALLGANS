from util.numpy_utils import np_imgs_NCWH_to_NHWC, np_index_to_onehot
from data_handler.BaseDataset import BaseDataset, DatasetCollection, DownloadInfo
from glob import glob
import numpy as np
import os
import pickle


def X_transform(x):
    x = np.reshape(x, [-1, 3, 32, 32])
    x = np_imgs_NCWH_to_NHWC(x)
    return x


def Y_transform(y):
    y = np_index_to_onehot(y)
    return y


class CIFAR10_train(BaseDataset):
    _PATTERN_TRAIN_FILE = "*/data_batch_*"
    _PKCL_KEY_TRAIN_DATA = b"data"
    _PKCL_KEY_TRAIN_LABEL = b"labels"
    LABEL_SIZE = 10

    @property
    def downloadInfos(self):
        return [
            DownloadInfo(
                url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                is_zipped=True,
                download_file_name='cifar-10-python.tar.gz',
                extracted_file_names=[
                    "data_batch_1",
                    "data_batch_2",
                    "data_batch_3",
                    "data_batch_4",
                    "data_batch_5",
                    # "test_batch",
                    "batches.meta"
                ]
            )
        ]

    def load(self, path, limit=None):
        # load train data
        files = glob(os.path.join(path, self._PATTERN_TRAIN_FILE), recursive=True)
        files.sort()
        for file in files:
            with open(file, 'rb') as fo:
                dict_ = pickle.load(fo, encoding='bytes')

            x = dict_[self._PKCL_KEY_TRAIN_DATA]
            self._append_data('Xs', x)

            label = dict_[self._PKCL_KEY_TRAIN_LABEL]
            self._append_data('Ys', label)

    def save(self):
        pass

    def preprocess(self):
        self.data['Xs'] = X_transform(self.data['Xs'])
        self.data['Ys'] = Y_transform(self.data['Ys'])


class CIFAR10_test(BaseDataset):
    _PATTERN_TEST_FILE = "*/test_batch"
    _PKCL_KEY_TEST_DATA = b"data"
    _PKCL_KEY_TEST_LABEL = b"labels"
    LABEL_SIZE = 10

    @property
    def downloadInfos(self):
        return [
            DownloadInfo(
                url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                is_zipped=True,
                download_file_name='cifar-10-python.tar.gz',
                extracted_file_names=[
                    # "data_batch_1",
                    # "data_batch_2",
                    # "data_batch_3",
                    # "data_batch_4",
                    # "data_batch_5",
                    "test_batch",
                    "batches.meta"
                ]
            )
        ]

    def load(self, path, limit=None):
        # load test data
        files = glob(os.path.join(path, self._PATTERN_TEST_FILE), recursive=True)
        files.sort()
        for file in files:
            with open(file, 'rb') as fo:
                dict_ = pickle.load(fo, encoding='bytes')

            x = dict_[self._PKCL_KEY_TEST_DATA]
            self._append_data('Xs', x)

            label = dict_[self._PKCL_KEY_TEST_LABEL]
            self._append_data('Ys', label)

    def save(self):
        pass

    def preprocess(self):
        self.data['Xs'] = X_transform(self.data['Xs'])
        self.data['Ys'] = Y_transform(self.data['Ys'])


class CIFAR10(DatasetCollection):
    def __init__(self, train_set=None, test_set=None, validation_set=None):
        super().__init__(train_set, test_set, validation_set)
        self.train_set = CIFAR10_train()
        self.test_set = CIFAR10_test()
