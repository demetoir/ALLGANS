from util.Logger import Logger
from util.misc_util import *
import traceback
import sys
import numpy as np
import os


def _check_attr_is_None(attr):
    def _check_attr_empty(f):
        def wrapper(self, *args):
            ret = f(self, *args)
            if getattr(self, attr) is None:
                raise ValueError("%s expect not None" % attr)
            return ret

        return wrapper

    return _check_attr_empty


class MetaTask(type):
    """Metaclass for hook inherited class's function
    metaclass ref from 'https://code.i-harness.com/ko/q/11fc307'
    """

    def __init__(cls, name, bases, clsdict):
        # add before, after task for AbstractDataset.load
        new_load = None
        if 'load' in clsdict:
            def new_load(self, path, limit):
                try:
                    if self.before_load_task is None:
                        self.if_need_download(path)
                    else:
                        self.before_load_task()

                    clsdict['load'](self, path, limit)

                    self.after_load(limit)
                except Exception:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    err_msg = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    self.log(*err_msg)

        setattr(cls, 'load', new_load)


class DownloadInfo:
    """download information for dataset
    self.url : download url
    self.is_zipped :
    self.zip_file_name:
    self.file_type :

    """

    def __init__(self, url, is_zipped=False, download_file_name=None, extracted_file_names=None):
        """create dataset download info

        :type url: str
        :type is_zipped: bool
        :type download_file_name: str
        :type extracted_file_names: list
        :param url: download url
        :param is_zipped: if zip file set True, else False
        :param download_file_name: file name of downloaded file
        :param extracted_file_names: file names of unzipped file
        """
        self.url = url
        self.is_zipped = is_zipped
        self.download_file_name = download_file_name
        self.extracted_file_names = extracted_file_names

    def attrs(self):
        return self.url, self.is_zipped, self.download_file_name, self.extracted_file_names


# todo may be dataset path in env_setting will be better, if automatically assign path as class name
class AbstractDataset(metaclass=MetaTask):
    """
    TODO
    """

    def __init__(self, preprocess=None, batch_after_task=None, before_load_task=None):
        """create dataset handler class

        ***bellow attrs must initiate other value after calling super()***
        self.download_infos: (list) dataset download info
        self.batch_keys: (str) feature label of dataset,
            managing batch keys in dict_keys.dataset_batch_keys recommend

        :param preprocess: injected function for preprocess dataset
        :param batch_after_task: function for inject into after iter mini_batch task
        :param before_load_task: function for injecting into AbstractDataset.before_load
        """
        self.download_infos = []
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

    def if_need_download(self, path):
        """check dataset is valid and if dataset is not valid download dataset

        :type path: str
        :param path: dataset path
        """
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

        for info in self.download_infos:
            is_Invalid = False
            files = glob(os.path.join(path, '**'), recursive=True)
            names = list(map(lambda file: os.path.split(file)[1], files))

            if info.is_zipped:
                file_list = info.extracted_file_names
                for data_file in file_list:
                    if data_file not in names:
                        is_Invalid = True
            else:
                if info.download_file_name not in names:
                    is_Invalid = True

            if is_Invalid:
                head, _ = os.path.split(path)
                download_file = os.path.join(path, info.download_file_name)

                self.log('download %s at %s ' % (info.download_file_name, download_file))
                download_from_url(info.url, download_file)

                if info.is_zipped:
                    self.log("extract %s at %s" % (info.download_file_name, path))
                    extract_file(download_file, path)

    def add_data(self, key, data):
        self.batch_keys += [key]
        self.data[key] = data
        self.cursor[key] = 0

    def get_data(self, keys):
        ret = [self.data[key] for key in keys]
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def after_load(self, limit=None):
        """after task for dataset and do execute preprocess for dataset

        init cursor for each batch_key
        limit dataset size
        execute preprocess

        :type limit: int
        :param limit: limit size of dataset
        """
        for key in self.batch_keys:
            self.cursor[key] = 0

        if limit is not None:
            for key in self.batch_keys:
                self.data[key] = self.data[key][:limit]

        for key in self.batch_keys:
            self.data_size = max(len(self.data[key]), self.data_size)
            self.log("batch data '%s' %d item(s) loaded" % (key, len(self.data[key])))
        self.log('%s fully loaded' % self.__str__())

        if self.preprocess is not None:
            self.preprocess(self)
            self.log('%s preprocess end' % self.__str__())

    # todo may be arg path default value is None and if None just feed default dataset path
    def load(self, path, limit=None):
        """
        TODO
        :param path:
        :param limit:
        :return:
        """
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
        """return iter mini batch

        ex)
        dataset.next_batch(3, ["train_x", "train_label"]) =
            [[train_x1, train_x2, train_x3], [train_label1, train_label2, train_label3]]

        dataset.next_batch(3, ["train_x", "train_label"], lookup=True) =
            [[train_x4, train_x5, train_x6], [train_label4, train_label5, train_label6]]

        dataset.next_batch(3, ["train_x", "train_label"]) =
            [[train_x4, train_x5, train_x6], [train_label4, train_label5, train_label6]]

        :param batch_size: size of mini batch
        :param batch_keys: (iterable type) select keys,
            if  batch_keys length is 1 than just return mini batch
            else return list of mini batch
        :param lookup: lookup == True cursor will not update
        :return: (numpy array type) list of mini batch, order is same with batch_keys
        """
        if batch_keys is None:
            batch_keys = self.batch_keys

        batches = []
        for key in batch_keys:
            batches += [self.__next_batch(batch_size, key, lookup)]

        if self.batch_after_task is not None:
            batches = self.batch_after_task(batches, batch_keys)

        if len(batches) == 1:
            batches = batches[0]

        return batches


class AbstractDatasetHelper:
    @staticmethod
    def preprocess(dataset):
        """preprocess for loaded data

        :param dataset: target dataset
        """
        raise NotImplementedError

    @staticmethod
    def next_batch_task(batch, batch_keys):
        """pre process for every iteration for mini batch

        * must return some mini batch

        :param batch: mini batch
        :return: batch
        """
        raise NotImplementedError

    @staticmethod
    def load_dataset(path, limit=None):
        """load dataset and return dataset and input_shapes of data

        * return value of input_shapes must contain every input_shape of data

        :type limit: int
        :param limit: limit number of dataset_size

        :return: dataset, input_shapes
        """
        raise NotImplementedError
