from util.Logger import Logger
import numpy as np


# TODO move this wrapper function some other way
def check_attr_empty(attr):
    def _check_attr_empty(f):
        def wrapper(self, *args):
            ret = f(self, *args)
            if len(getattr(self, attr)) == 0:
                raise ValueError("%s has empty" % attr)
            return ret

        return wrapper

    return _check_attr_empty


class AbstractDataset:
    def __init__(self, preprocess=None, batch_after_task=None):
        self.logger = Logger(self.__class__.__name__, stdout_only=True)
        self.log = self.logger.get_log()
        self.preprocess = preprocess
        self.batch_after_task = batch_after_task
        self.data = {}
        self.cursor = {}
        self.shape = None
        self.batch_keys = None
        self.data_size = 0

    def __del__(self):
        del self.shape
        del self.data
        del self.cursor
        del self.logger
        del self.log
        del self.batch_after_task
        del self.batch_keys

    def __repr__(self):
        return self.__class__.__name__

    @check_attr_empty("data")
    @check_attr_empty("cursor")
    def load(self, path, limit=None):
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
