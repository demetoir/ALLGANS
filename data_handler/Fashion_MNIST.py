from data_handler.AbstractDataset import AbstractDataset
from tensorflow.examples.tutorials.mnist import input_data
from env_settting import FASHION_MNIST_PATH
from dict_keys.dataset_batch_keys import *
from dict_keys.input_shape_keys import *
import numpy as np


class Fashion_MNIST(AbstractDataset):
    LABEL_SIZE = 10
    TRAIN_SIZE = 60000
    TEST_SIZE = 10000

    def __init__(self, preprocess=None, batch_after_task=None):
        super().__init__(preprocess, batch_after_task)
        self.batch_keys = [
            BATCH_KEY_TRAIN_X,
            BATCH_KEY_TRAIN_LABEL,
            BATCH_KEY_TEST_X,
            BATCH_KEY_TEST_LABEL
        ]
        self._SOURCE_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
        self._SOURCE_FILE = "..."

        def dummy():
            pass

        self.before_load_task = dummy

    def load(self, path, limit=None):
        # fashion_mnist = input_data.read_data_sets(path, one_hot=True)
        fashion_mnist = input_data.read_data_sets(path,
                                                  source_url=self._SOURCE_URL,
                                                  one_hot=True)
        # load train data
        train_x, train_label = fashion_mnist.train.next_batch(self.TRAIN_SIZE)
        self.data[BATCH_KEY_TRAIN_X] = train_x
        self.data[BATCH_KEY_TRAIN_LABEL] = train_label

        # load test data
        test_x, test_label = fashion_mnist.test.next_batch(self.TEST_SIZE)
        self.data[BATCH_KEY_TEST_X] = test_x
        self.data[BATCH_KEY_TEST_LABEL] = test_label

    def save(self):
        raise NotImplementedError


class Fashion_MNISTHelper:
    @staticmethod
    def preprocess(dataset):
        # original fashion_mnist image size is 28*28 but need to resize 32*32
        data = dataset.data[BATCH_KEY_TRAIN_X]
        shape = data.shape
        data = np.reshape(data, [shape[0], 28, 28])
        npad = [(0, 0), (2, 2), (2, 2)]
        data = np.pad(data, pad_width=npad, mode='constant', constant_values=0.0)
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
