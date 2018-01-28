from data_handler.AbstractDataset import AbstractDataset
from glob import glob
from dict_keys.dataset_batch_keys import *
import os
import pickle
import requests
import tarfile

from util.util import extract_tar, download_data


class CIFAR10(AbstractDataset):
    PATTERN_TRAIN_FILE = "data_batch_*"
    PATTERN_TEST_FILE = "test_batch"
    PKCL_KEY_TRAIN_DATA = b"data"
    PKCL_KEY_TRAIN_LABEL = b"labels"
    PKCL_KEY_TEST_DATA = b"data"
    PKCL_KEY_TEST_LABEL = b"labels"
    LABEL_SIZE = 10
    SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    SOURCE_FILE = 'cifar-10-python.tar.gz'

    def __init__(self, preprocess=None, batch_after_task=None):
        super().__init__(preprocess, batch_after_task)
        self.batch_keys = [BATCH_KEY_TRAIN_X, BATCH_KEY_TRAIN_LABEL, BATCH_KEY_TEST_X, BATCH_KEY_TEST_LABEL]

    def load(self, path, limit=None):
        try:
            os.makedirs(path)
            self.log('create dir %s' % path)
        except FileExistsError:
            pass

        # TODO hack better way to inspect data file
        download = False
        files = glob(os.path.join(path, self.PATTERN_TRAIN_FILE))
        if len(files) != 5:
            download = True

        files = glob(os.path.join(path, self.PATTERN_TEST_FILE))
        if len(files) != 1:
            download = True

        if download:
            head, _ = os.path.split(path)
            download_file = os.path.join(head, self.SOURCE_FILE)
            self.log('download %s at %s ' % (self.SOURCE_FILE, download_file))
            download_data(self.SOURCE_URL, download_file)

            self.log("extract %s at %s" % (self.SOURCE_FILE, head))
            extract_tar(download_file, head)

        try:
            # load train data
            files = glob(os.path.join(path, self.PATTERN_TRAIN_FILE))
            files.sort()
            for file in files:
                with open(file, 'rb') as fo:
                    dict_ = pickle.load(fo, encoding='bytes')

                x = dict_[self.PKCL_KEY_TRAIN_DATA]
                self._append_data(BATCH_KEY_TRAIN_X, x)

                label = dict_[self.PKCL_KEY_TRAIN_LABEL]
                self._append_data(BATCH_KEY_TRAIN_LABEL, label)

            # load test data
            files = glob(os.path.join(path, self.PATTERN_TEST_FILE))
            files.sort()
            for file in files:
                with open(file, 'rb') as fo:
                    dict_ = pickle.load(fo, encoding='bytes')

                x = dict_[self.PKCL_KEY_TEST_DATA]
                self._append_data(BATCH_KEY_TEST_X, x)

                label = dict_[self.PKCL_KEY_TEST_LABEL]
                self._append_data(BATCH_KEY_TEST_LABEL, label)

        except Exception as e:
            self.log(e)

        super().load(path, limit)

    def save(self):
        raise NotImplementedError
