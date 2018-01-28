from data_handler.CIFAR10 import CIFAR10
from env_settting import CIFAR10_PATH
from util.util import np_index_to_onehot, np_img_NCWH_to_NHWC
from dict_keys.dataset_batch_keys import *
from dict_keys.input_shape_keys import *
import numpy as np


class cifar10Helper:
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
        data = np_img_NCWH_to_NHWC(data)
        dataset.data[BATCH_KEY_TRAIN_X] = data

        data = dataset.data[BATCH_KEY_TEST_X]
        data = np.reshape(data, [-1, 3, 32, 32])
        data = np_img_NCWH_to_NHWC(data)
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
        cifar10 = CIFAR10(preprocess=cifar10Helper.preprocess)
        cifar10.load(CIFAR10_PATH, limit=limit)
        input_shapes = {
            INPUT_SHAPE_KEY_DATA_X: [32, 32, 3],
            INPUT_SHAPE_KEY_LABEL: [10],
            INPUT_SHAPE_KEY_LABEL_SIZE: 10
        }
        return cifar10, input_shapes
