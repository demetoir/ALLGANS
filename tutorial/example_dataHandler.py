"""example of implementing data handler

using MNIST dataset

implement step
1. add dataset folder path in env_setting.py

    EXAMPLE_DATASET_PATH = os.path.join(DATA_PATH, 'example_dataset')

2. add dataset_batch_keys in dict_keys.dataset_batch_keys.py

3. add input_shapes_keys in dict_keys.dataset_batch_keys.py



2. implement dataset class in data_handler.dataset_name.py
    1. implement self.__init__

    2. implement self.load()

    3. implement self.

3. implement datasetHelper class in data_handler.dataset_name.py
    implement self.preprocess()

    implement self.next_batch_task()

    implement self.load_dataset()

"""
from tensorflow.examples.tutorials.mnist import input_data
from data_handler.AbstractDataset import AbstractDataset, AbstractDatasetHelper
from dict_keys.dataset_batch_keys import BATCH_KEY_TRAIN_X, BATCH_KEY_TRAIN_LABEL, BATCH_KEY_TEST_X, \
    BATCH_KEY_TEST_LABEL
import numpy as np

from dict_keys.input_shape_keys import INPUT_SHAPE_KEY_DATA_X, INPUT_SHAPE_KEY_LABEL, INPUT_SHAPE_KEY_LABEL_SIZE
from env_settting import EXAMPLE_DATASET_PATH


class ExampleDataset(AbstractDataset):
    # need to child class of AbstractDataset

    TRAIN_SIZE = 60000
    TEST_SIZE = 10000
    LABEL_SIZE = 10

    def __init__(self, preprocess=None, batch_after_task=None):
        # always call super() first
        super().__init__(preprocess, batch_after_task)
        # always set batch_keys to load each data
        self.batch_keys = [BATCH_KEY_TRAIN_X, BATCH_KEY_TRAIN_LABEL, BATCH_KEY_TEST_X, BATCH_KEY_TEST_LABEL]

        def dummy():
            pass

        # if no need to download assign dummy function to self.if_need_download
        self.if_need_download = dummy

    def load(self, path, limit=None):
        # mnist is loading by tensorflow built-in
        mnist = input_data.read_data_sets(path, one_hot=True)

        train_x, train_label = mnist.train.next_batch(self.TRAIN_SIZE)
        test_x, test_label = mnist.test.next_batch(self.TEST_SIZE)

        # make dict with batch key : data
        self.data = {
            BATCH_KEY_TRAIN_X: train_x,
            BATCH_KEY_TRAIN_LABEL: train_label,
            BATCH_KEY_TEST_X: test_x,
            BATCH_KEY_TEST_LABEL: test_label
        }


class ExampleDatasetHelper(AbstractDatasetHelper):
    # need to child class of AbstractDatasetHelper
    @staticmethod
    def preprocess(dataset):
        # you can implement preprocess here for loaded dataset
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
        # implement next_batch_task for every iteration of mini batch
        # if no need to implement next_batch_task, just return batch
        x, label = batch[0], batch[1]
        return x, label

    @staticmethod
    def load_dataset(limit=None):
        # implement helper for load_dataset
        # must return loaded dataset and dataset's input_shapes
        # input_shapes is dict.
        # key is input_shape_key in dict_keys.input_shape_keys
        # value numpy shape of data
        mnist_data = ExampleDataset(preprocess=ExampleDatasetHelper.preprocess)
        mnist_data.load(EXAMPLE_DATASET_PATH, limit=limit)
        input_shapes = {
            INPUT_SHAPE_KEY_DATA_X: [32, 32, 1],
            INPUT_SHAPE_KEY_LABEL: [10],
            INPUT_SHAPE_KEY_LABEL_SIZE: 10,
        }

        return mnist_data, input_shapes
