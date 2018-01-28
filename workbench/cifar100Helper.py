from data_handler.CIFAR100 import CIFAR100
from env_settting import CIFAR100_PATH
from dict_keys.dataset_batch_keys import *
import numpy as np


class cifar100Helper:
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
    def load_dataset():
        cifar100 = CIFAR100(preprocess=cifar100Helper.preprocess)
        cifar100.load(CIFAR100_PATH)
        return cifar100, [32, 32, 3]
