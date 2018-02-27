from data_handler.AbstractDataset import AbstractDataset, AbstractDatasetHelper
from env_settting import CIFAR10_PATH
from util.numpy_utils import np_imgs_NCWH_to_NHWC, np_index_to_onehot
from dict_keys.dataset_batch_keys import *
from dict_keys.input_shape_keys import *
from glob import glob
from data_handler.AbstractDataset import DownloadInfo
import numpy as np
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
        self.batch_keys = [
            BATCH_KEY_TRAIN_X,
            BATCH_KEY_TRAIN_LABEL,
            BATCH_KEY_TEST_X,
            BATCH_KEY_TEST_LABEL]

        self.download_infos = [
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
                    "test_batch",
                    "batches.meta"
                ]
            )
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


class CIFAR10Helper(AbstractDatasetHelper):
    @staticmethod
    def next_batch_task(batch):
        x = batch[0]
        label = batch[1]
        return x, label

    @staticmethod
    def preprocess(dataset):
        # convert image format from NCWH to NHWC
        data = dataset.data[BATCH_KEY_TRAIN_X]
        data = np.reshape(data, [-1, 3, 32, 32])
        data = np_imgs_NCWH_to_NHWC(data)
        dataset.data[BATCH_KEY_TRAIN_X] = data

        data = dataset.data[BATCH_KEY_TEST_X]
        data = np.reshape(data, [-1, 3, 32, 32])
        data = np_imgs_NCWH_to_NHWC(data)
        dataset.data[BATCH_KEY_TEST_X] = data

        # make label index to onehot
        data = dataset.data[BATCH_KEY_TRAIN_LABEL]
        data = np_index_to_onehot(data)
        dataset.data[BATCH_KEY_TRAIN_LABEL] = data

        data = dataset.data[BATCH_KEY_TEST_LABEL]
        data = np_index_to_onehot(data)
        dataset.data[BATCH_KEY_TEST_LABEL] = data

    @staticmethod
    def load_dataset(limit=None):
        cifar10 = CIFAR10(preprocess=CIFAR10Helper.preprocess)
        cifar10.load(CIFAR10_PATH, limit=limit)
        input_shapes = {
            INPUT_SHAPE_KEY_DATA_X: [32, 32, 3],
            INPUT_SHAPE_KEY_LABEL: [10],
            INPUT_SHAPE_KEY_LABEL_SIZE: 10
        }
        return cifar10, input_shapes
