from data_handler.AbstractDataset import AbstractDataset
import os
import pickle


class CIFAR100(AbstractDataset):
    TRAIN_x = "train_data"
    TRAIN_fine_labels = "train_fine_labels"
    TRAIN_coarse_labels = "train_coarse_labels"

    TEST_x = "test_data"
    TEST_fine_labels = "test_fine_labels"
    TEST_coarse_labels = "test_coarse_labels"

    PATTERN_TRAIN_FILE = "train"
    PATTERN_TEST_FILE = "test"

    pkcl_key_train_data = b"data"
    pkcl_key_train_fine_labels = b"fine_labels"
    pkcl_key_train_coarse_labels = b"coarse_labels"

    pkcl_key_test_data = b"data"
    pkcl_key_test_fine_labels = b"fine_labels"
    pkcl_key_test_coarse_labels = b"coarse_labels"

    def __init__(self, preprocess=None, batch_after_task=None):
        super().__init__(preprocess, batch_after_task)
        self.keys = [self.TRAIN_x, self.TRAIN_coarse_labels, self.TRAIN_fine_labels, self.TEST_x,
                     self.TEST_coarse_labels, self.TEST_fine_labels]

    def load(self, path, limit=None):
        try:
            # load train data
            file_path = os.path.join(path, self.PATTERN_TRAIN_FILE)
            with open(file_path, 'rb') as fo:
                dict_ = pickle.load(fo, encoding='bytes')
            x = dict_[self.pkcl_key_train_data]
            coarse_labels = dict_[self.pkcl_key_train_coarse_labels]
            fine_labels = dict_[self.pkcl_key_train_fine_labels]
            self._append_data(self.TRAIN_x, x)
            self._append_data(self.TRAIN_coarse_labels, coarse_labels)
            self._append_data(self.TRAIN_fine_labels, fine_labels)

            # load test data
            file_path = os.path.join(path, self.PATTERN_TEST_FILE)
            with open(file_path, 'rb') as fo:
                dict_ = pickle.load(fo, encoding='bytes')
            x = dict_[self.pkcl_key_test_data]
            coarse_labels = dict_[self.pkcl_key_test_coarse_labels]
            fine_labels = dict_[self.pkcl_key_test_fine_labels]
            self._append_data(self.TEST_x, x)
            self._append_data(self.TEST_coarse_labels, coarse_labels)
            self._append_data(self.TEST_fine_labels, fine_labels)

        except Exception as e:
            self.log(e)

        super().load(path, limit)

    def save(self):
        raise NotImplementedError
