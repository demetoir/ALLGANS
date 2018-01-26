from data_handler.Fashion_MNIST import Fashion_MNIST
from env_settting import FASHION_MNIST_PATH
from util.util import np_index_to_onehot
from dict_keys.dataset_batch_keys import *
from dict_keys.input_shape_keys import *
import numpy as np


class fashion_MNISTHelper:
    @staticmethod
    def preprocess(dataset):
        # original fashion_mnist image size is 28*28 but need to resize 32*32
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

        # label index to onehot
        data = dataset.data[BATCH_KEY_TRAIN_LABEL]
        data = np.reshape(data, [len(data)])
        data = data.astype(np.int32)
        data = np_index_to_onehot(data, Fashion_MNIST.LABEL_SIZE)
        dataset.data[BATCH_KEY_TRAIN_LABEL] = data

        data = dataset.data[BATCH_KEY_TEST_LABEL]
        data = np.reshape(data, [len(data)])
        data = data.astype(np.int32)
        data = np_index_to_onehot(data, Fashion_MNIST.LABEL_SIZE)
        dataset.data[BATCH_KEY_TEST_LABEL] = data

    @staticmethod
    def load_dataset(limit=None):
        dataset = Fashion_MNIST(preprocess=fashion_MNISTHelper.preprocess)
        dataset.load(FASHION_MNIST_PATH, limit=limit)
        input_shapes = {
            INPUT_SHAPE_KEY_DATA_X: [32, 32, 1],
            INPUT_SHAPE_KEY_LABEL: [10],
            INPUT_SHAPE_KEY_LABEL_SIZE: 10,
        }

        return dataset, input_shapes

    @staticmethod
    def next_batch_task(batch):
        x = batch[0]
        return x
