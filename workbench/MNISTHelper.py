from data_handler.MNIST import MNIST
from env_settting import MNIST_PATH
from dict_keys.dataset_batch_keys import *
from dict_keys.input_shape_keys import *
import numpy as np


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
