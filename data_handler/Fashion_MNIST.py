from data_handler.AbstractDataset import AbstractDataset
from tensorflow.examples.tutorials.mnist import input_data
from dict_keys.dataset_batch_keys import *


class Fashion_MNIST(AbstractDataset):
    SOURCE_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    LABEL_SIZE = 10
    TRAIN_SIZE = 60000
    TEST_SIZE = 10000

    def __init__(self, preprocess=None, batch_after_task=None):
        super().__init__(preprocess, batch_after_task)
        self.batch_keys = [BATCH_KEY_TRAIN_X, BATCH_KEY_TRAIN_LABEL, BATCH_KEY_TEST_X, BATCH_KEY_TEST_LABEL]

    def load(self, path, limit=None):
        # fashion_mnist = input_data.read_data_sets(path, one_hot=True)
        fashion_mnist = input_data.read_data_sets(path,
                                                  source_url=self.SOURCE_URL,
                                                  one_hot=True)
        # load train data
        train_x, train_label = fashion_mnist.train.next_batch(self.TRAIN_SIZE)
        self.data[BATCH_KEY_TRAIN_X] = train_x
        self.data[BATCH_KEY_TRAIN_LABEL] = train_label

        # load test data
        test_x, test_label = fashion_mnist.test.next_batch(self.TEST_SIZE)
        self.data[BATCH_KEY_TEST_X] = test_x
        self.data[BATCH_KEY_TEST_LABEL] = test_label

        super().load(path, limit)

    def save(self):
        raise NotImplementedError
