from util.Logger import Logger
from glob import glob
from util.util import *
import traceback
import sys
import numpy as np
import os


def check_attr_is_None(attr):
    def _check_attr_empty(f):
        def wrapper(self, *args):
            ret = f(self, *args)
            if getattr(self, attr) is None:
                raise ValueError("%s expect not None" % attr)
            return ret

        return wrapper

    return _check_attr_empty


class MetaTask(type):
    """
    metaclass ref from 'https://code.i-harness.com/ko/q/11fc307'

    """

    def __init__(cls, name, bases, clsdict):
        # add before, after task for AbstractDataset.load
        if 'load' in clsdict:
            def new_load(self, path, limit):
                try:
                    if self.before_load_task is None:
                        self.before_load(path)
                    else:
                        self.before_load_task()

                    clsdict['load'](self, path, limit)

                    self.after_load(limit)
                except Exception as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    err_msg = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    self.log(*err_msg)

        setattr(cls, 'load', new_load)

        # check subclass init value
        # if '__init__' in clsdict:
        #     def new__init__(self, *args, **kwargs):
        #         clsdict['__init__'](self, *args, **kwargs)
        #             def is_None(name, value):
        #                 if value is None:
        #                     raise ValueError("%s expect not None" % name)
        #         is_None('self._SOURCE_URL', self._SOURCE_URL)
        #         is_None('self._SOURCE_FILE', self._SOURCE_FILE)
        #         is_None('self._data_files', self._data_files)
        #         is_None('self.batch_keys', self.batch_keys)
        #
        # setattr(cls, '__init__', new__init__)


class AbstractDataset(metaclass=MetaTask):
    def __init__(self, preprocess=None, batch_after_task=None, before_load_task=None):
        """
        init dataset attrs

        *** bellow attrs must initiate other value ***
        self._SOURCE_URL: (str) url for download dataset
        self._SOURCE_FILE: (str) file name of zipped dataset
        self._data_files = (str) files name in dataset
        self.batch_keys = (str) feature label of dataset,
            managing batch keys in dict_keys.dataset_batch_keys recommend

        :param preprocess: injected function for preprocess dataset
        :param batch_after_task: injected function for after iter mini_batch
        :param before_load_task: hookable function for AbstractDataset.before_load
        """
        self._SOURCE_URL = None
        self._SOURCE_FILE = None
        self._data_files = None
        self.batch_keys = None
        self.logger = Logger(self.__class__.__name__, stdout_only=True)
        self.log = self.logger.get_log()
        self.preprocess = preprocess
        self.batch_after_task = batch_after_task
        self.data = {}
        self.cursor = {}
        self.data_size = 0
        self.before_load_task = before_load_task

    def __del__(self):
        del self.data
        del self.cursor
        del self.logger
        del self.log
        del self.batch_after_task
        del self.batch_keys

    def __repr__(self):
        return self.__class__.__name__

    def before_load(self, path):
        """
        check dataset is valid and if dataset is not valid download dataset

        :param path: dataset path
        :return:
        """
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

        is_Invalid = False
        files = glob(os.path.join(path, '*'))
        names = list(map(lambda file: os.path.split(file)[1], files))
        for data_file in self._data_files:
            if data_file not in names:
                is_Invalid = True

        if is_Invalid:
            head, _ = os.path.split(path)
            download_file = os.path.join(head, self._SOURCE_FILE)
            self.log('download %s at %s ' % (self._SOURCE_FILE, download_file))
            download_data(self._SOURCE_URL, download_file)

            self.log("extract %s at %s" % (self._SOURCE_FILE, head))
            extract_data(download_file, head)

    def after_load(self, limit=None):
        """
        after task for dataset and do execute preprocess for dataset

        init cursor for each batch_key
        limit dataset size
        execute preprocess

        :param limit: limit size of dataset
        :return:
        """
        for key in self.batch_keys:
            self.cursor[key] = 0

        for key in self.batch_keys:
            self.data[key] = self.data[key][:limit]

        for key in self.batch_keys:
            self.data_size = max(len(self.data[key]), self.data_size)
            self.log("batch data '%s' %d item(s) loaded" % (key, len(self.data[key])))
        self.log('%s fully loaded' % self.__str__())

        if self.preprocess is not None:
            self.preprocess(self)
            self.log('%s preprocess end' % self.__str__())

    def load(self, path, limit=None):
        pass

    def save(self):
        raise NotImplementedError

    def _append_data(self, batch_key, data):
        if batch_key not in self.data:
            self.data[batch_key] = data
        else:
            self.data[batch_key] = np.concatenate((self.data[batch_key], data))

    def __next_batch(self, batch_size, key, lookup=False):
        data = self.data[key]
        cursor = self.cursor[key]
        data_size = len(data)

        # if batch size exceeds the size of data set
        over_data = batch_size // data_size
        if over_data > 0:
            whole_data = np.concatenate((data[cursor:], data[:cursor]))
            batch_to_append = np.repeat(whole_data, over_data, axis=0)
            batch_size -= data_size * over_data
        else:
            batch_to_append = None

        begin, end = cursor, (cursor + batch_size) % data_size

        if begin < end:
            batch = data[begin:end]
        else:
            first, second = data[begin:], data[:end]
            batch = np.concatenate((first, second))

        if batch_to_append:
            batch = np.concatenate((batch_to_append, batch))

        if not lookup:
            self.cursor[key] = end

        return batch

    def next_batch(self, batch_size, batch_keys=None, lookup=False):
        """
        return iter mini batch

        :param batch_size: size of mini batch
        :param batch_keys: (iterable type) select keys,
            if  batch_keys length is 1 than just return mini batch
            else return list of mini batch
        :param lookup: lookup == True cursor will not update
        :return: (numpy array type) list of mini batch, order is same with batch_keys



        ex)
        dataset.next_batch(3, ["train_x", "train_label"]) =
            [[train_x1, train_x2, train_x3], [train_label1, train_label2, train_label3]]

        dataset.next_batch(3, ["train_x", "train_label"], lookup=True) =
            [[train_x4, train_x5, train_x6], [train_label4, train_label5, train_label6]]

        dataset.next_batch(3, ["train_x", "train_label"]) =
            [[train_x4, train_x5, train_x6], [train_label4, train_label5, train_label6]]

        """
        if batch_keys is None:
            batch_keys = self.batch_keys

        batches = []
        for key in batch_keys:
            batches += [self.__next_batch(batch_size, key, lookup)]

        if self.batch_after_task is not None:
            batches = self.batch_after_task(batches)

        if len(batches) == 1:
            batches = batches[0]

        return batches
