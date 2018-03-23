from util.Logger import Logger
from util.misc_util import *
import traceback
import sys
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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

    def __init__(cls, name, bases, cls_dict):
        # hook if_need_download, after_load for AbstractDataset.load
        new_load = None
        if 'load' in cls_dict:
            def new_load(self, path, limit):
                try:
                    self.if_need_download(path)
                    cls_dict['load'](self, path, limit)
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


class BaseDataset(metaclass=MetaTask):
    """
    TODO
    """

    def __init__(self):
        """create dataset handler class

        ***bellow attrs must initiate other value after calling super()***
        self.download_infos: (list) dataset download info
        self.batch_keys: (str) feature label of dataset,
            managing batch keys in dict_keys.dataset_batch_keys recommend

        """
        self.download_infos = []
        self.batch_keys = []
        self.logger = Logger(self.__class__.__name__, stdout_only=True)
        self.log = self.logger.get_log()
        self.data = {}
        self.cursor = 0
        self.data_size = 0
        self.input_shapes = None

    def __del__(self):
        del self.logger
        del self.log

    def __repr__(self):
        return self.__class__.__name__

    def add_data(self, key, data):
        self.data[key] = data
        self.cursor = 0
        self.data_size = len(data)

    def get_data(self, key):
        return self.data[key]

    def get_datas(self, keys):
        return [self.data[key] for key in keys]

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
            if self.is_invalid(path, info):
                self.download_data(path, info)

    def is_invalid(self, path, download_info):
        validation = None
        files = glob(os.path.join(path, '**'), recursive=True)
        names = list(map(lambda file: os.path.split(file)[1], files))

        if download_info.is_zipped:
            file_list = download_info.extracted_file_names
            for data_file in file_list:
                if data_file not in names:
                    validation = True
        else:
            if download_info.download_file_name not in names:
                validation = True

        return validation

    def download_data(self, path, download_info):
        head, _ = os.path.split(path)
        download_file = os.path.join(path, download_info.download_file_name)

        self.log('download %s at %s ' % (download_info.download_file_name, download_file))
        download_from_url(download_info.url, download_file)

        if download_info.is_zipped:
            self.log("extract %s at %s" % (download_info.download_file_name, path))
            extract_file(download_file, path)

    def after_load(self, limit=None):
        """after task for dataset and do execute preprocess for dataset

        init cursor for each batch_key
        limit dataset size
        execute preprocess

        :type limit: int
        :param limit: limit size of dataset
        """

        if limit is not None:
            for key in self.batch_keys:
                self.data[key] = self.data[key][:limit]

        for key in self.data:
            self.data_size = max(len(self.data[key]), self.data_size)
            self.log("batch data '%s' %d item(s) loaded" % (key, len(self.data[key])))

        self.log('%s fully loaded' % self.__str__())

        self.log('%s preprocess end' % self.__str__())
        self.preprocess()

        self.log("generate input_shapes")
        self.input_shapes = {}
        for key in self.data:
            self.input_shapes[key] = list(self.data[key].shape[1:])
            self.log(key, self.input_shapes[key])

    def load(self, path, limit=None):
        """
        TODO
        :param path:
        :param limit:
        :return:
        """
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def _append_data(self, batch_key, data):
        if batch_key not in self.data:
            self.data[batch_key] = data
        else:
            self.data[batch_key] = np.concatenate((self.data[batch_key], data))

    def iter_batch(self, data, batch_size):
        cursor = self.cursor
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

        return batch

    def next_batch(self, batch_size, batch_keys=None, look_up=False):

        if batch_keys is None:
            batch_keys = self.batch_keys

        batches = []
        for key in batch_keys:
            batches += [self.iter_batch(self.data[key], batch_size)]

        if not look_up:
            self.cursor = (self.cursor + batch_size) % self.data_size

        batches = self.after_next_batch(batches, batch_keys)

        return batches[0] if len(batches) == 1 else batches

    def preprocess(self):
        """preprocess for loaded data

        """
        raise NotImplementedError

    def after_next_batch(self, batches, batch_keys=None):
        """pre process for every iteration for mini batch

        * must return some mini batch

        :param batch_keys:
        :param batches:
        :return: batch
        """
        return batches

    def split(self, ratio, shuffle=False):
        a_set = BaseDataset()
        b_set = BaseDataset()
        a_set.input_shapes = self.input_shapes
        b_set.input_shapes = self.input_shapes

        a_ratio = ratio[0] / sum(ratio)
        index = int(self.data_size * a_ratio)
        for key in self.data:
            a_set.add_data(key, self.data[key][:index])
            b_set.add_data(key, self.data[key][index:])

        return a_set, b_set

    def merge(self, a_set, b_set):
        new_set = BaseDataset()
        if a_set.keys() is not b_set.keys():
            raise KeyError("dataset can not merge, key does not match")

        for key in a_set:
            new_set.add_data(key, a_set.data[key] + b_set.data[key])

        return new_set

    def shuffle(self):
        random_state = np.random.randint(1, 12345678)

        for key in self.batch_keys:
            self.data[key] = shuffle(self.data[key], random_state=random_state)


class DatasetCollection:
    def __init__(self):
        self.train_set = None
        self.test_set = None
        self.validation_set = None

    def load(self, path, **kwargs):
        if self.train_set is not None:
            self.train_set.load(path, **kwargs)

        if self.test_set is not None:
            self.test_set.load(path, **kwargs)

        if self.validation_set is not None:
            self.validation_set.load(path, **kwargs)

    @property
    def input_shapes(self):
        raise NotImplementedError
