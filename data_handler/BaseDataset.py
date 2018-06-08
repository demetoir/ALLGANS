from util.Logger import StdoutOnlyLogger
from util.misc_util import *
import traceback
import sys
import numpy as np
import os
import sklearn.utils


def _check_attr_is_None(attr):
    def _check_attr_empty(f):
        def wrapper(self, *args):
            ret = f(self, *args)
            if getattr(self, attr) is None:
                raise ValueError("%s expect not None" % attr)
            return ret

        return wrapper

    return _check_attr_empty


class MetaDataset(type):
    """Metaclass for hook inherited class's function
    metaclass ref from 'https://code.i-harness.com/ko/q/11fc307'
    """

    def __init__(cls, name, bases, cls_dict):
        type.__init__(cls, name, bases, cls_dict)

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


class BaseDataset(metaclass=MetaDataset):
    """
    TODO add docstring
    """

    @property
    def downloadInfos(self):
        return []

    def __init__(self):
        """create dataset handler class

        ***bellow attrs must initiate other value after calling super()***
        self.download_infos: (list) dataset download info
        self.batch_keys: (str) feature label of dataset,
            managing batch keys in dict_keys.dataset_batch_keys recommend
        """
        self.batch_keys = []
        self.logger = StdoutOnlyLogger(self.__class__.__name__)
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
        self.batch_keys += [key]

    def get_data(self, key):
        """return data

        :param key:
        :type key: str

        :return:
        """
        return self.data[key]

    def get_datas(self, keys):
        """return list of data

        :param keys: list of keys
        :type keys: list
        :return: list
        """
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

        for info in self.downloadInfos:
            if self._is_invalid(path, info):
                self.download_data(path, info)

    def _is_invalid(self, path, downloadInfos):
        """check dataset file validation"""
        validation = None
        files = glob(os.path.join(path, '**'), recursive=True)
        names = list(map(lambda file: os.path.split(file)[1], files))

        if downloadInfos.is_zipped:
            file_list = downloadInfos.extracted_file_names
            for data_file in file_list:
                if data_file not in names:
                    validation = True
        else:
            if downloadInfos.download_file_name not in names:
                validation = True

        return validation

    def download_data(self, path, downloadInfos):
        """donwnload data if need

        :param path:
        :param downloadInfos:
        :return:
        """
        head, _ = os.path.split(path)
        download_file = os.path.join(path, downloadInfos.download_file_name)

        self.log('download %s at %s ' % (downloadInfos.download_file_name, download_file))
        download_from_url(downloadInfos.url, download_file)

        if downloadInfos.is_zipped:
            self.log("extract %s at %s" % (downloadInfos.download_file_name, path))
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
            self.log("key=%s, shape=%s" % (key, self.input_shapes[key]))

    def load(self, path, limit=None):
        """load dataset from file should implement

        save data at self.data, expect dict type

        :param path:
        :param limit:
        :return: None
        """
        raise NotImplementedError

    def save(self):
        """

        :return: None
        """
        raise NotImplementedError

    def _append_data(self, batch_key, data):
        if batch_key not in self.data:
            self.data[batch_key] = np.array(data)
        else:
            self.data[batch_key] = np.concatenate((self.data[batch_key], data))

    def _iter_batch(self, data, batch_size):
        cursor = self.cursor
        data_size = len(data)

        # if batch size exceeds the size of data set
        over_data = batch_size // (data_size + 1)
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
        :param look_up: lookup == True cursor will not update
        :return: (numpy array type) list of mini batch, order is same with batch_keys
        """

        if batch_keys is None:
            batch_keys = self.batch_keys

        if type(batch_keys) is str:
            batch_keys = [batch_keys]

        batches = []
        for key in batch_keys:
            batches += [self._iter_batch(self.data[key], batch_size)]

        if not look_up:
            self.cursor = (self.cursor + batch_size) % self.data_size

        return batches[0] if len(batches) == 1 else batches

    def preprocess(self):
        """preprocess for loaded data

        """
        raise NotImplementedError

    def split(self, ratio, shuffle=False):
        """return split part of dataset"""
        a_set = self.__class__()
        b_set = self.__class__()
        a_set.input_shapes = self.input_shapes
        b_set.input_shapes = self.input_shapes

        a_ratio = ratio[0] / sum(ratio)
        index = int(self.data_size * a_ratio)
        for key in self.data:
            a_set.add_data(key, self.data[key][:index])
            b_set.add_data(key, self.data[key][index:])

        a_set.batch_keys = self.data.keys()
        b_set.batch_keys = self.data.keys()
        if shuffle:
            a_set.shuffle()
            b_set.shuffle()

        return a_set, b_set

    def merge(self, a_set, b_set):
        """merge to dataset"""
        if set(a_set.batch_keys) is set(b_set.batch_keys):
            raise KeyError("dataset can not merge, key does not match")

        new_set = a_set.__class__()
        for key in a_set.batch_keys:
            concated = np.concatenate((a_set.data[key], b_set.data[key]), axis=0)
            new_set.add_data(key, concated)

        return new_set

    def shuffle(self, random_state=None):
        """shuffle dataset"""
        if random_state is None:
            random_state = np.random.randint(1, 12345678)

        for key in self.data:
            self.data[key] = sklearn.utils.shuffle(self.data[key], random_state=random_state)

    def reset_cursor(self):
        """reset cursor"""
        self.cursor = 0

    def full_batch(self, batch_keys=None):
        if batch_keys is None:
            batch_keys = self.batch_keys

        if type(batch_keys) is str:
            batch_keys = [batch_keys]

        batches = []
        for key in batch_keys:
            batches += [self.data[key]]

        return batches[0] if len(batches) == 1 else batches

    @property
    def size(self):
        size = 0
        for key in self.data:
            size = max(len(self.data[key]), size)
        return size


class DatasetCollection:
    def __init__(self, train_set=None, test_set=None, validation_set=None):
        self.logger = StdoutOnlyLogger(self.__class__.__name__)
        self.log = self.logger.get_log()

        self.train_set = train_set
        self.test_set = test_set
        self.validation_set = validation_set

    def load(self, path, **kwargs):
        if self.train_set is not None:
            self.train_set.load(path, **kwargs)

        if self.test_set is not None:
            self.test_set.load(path, **kwargs)

        if self.validation_set is not None:
            self.validation_set.load(path, **kwargs)

    def shuffle(self):
        if self.train_set is not None:
            self.train_set.shuffle()
        if self.test_set is not None:
            self.test_set.shuffle()
        if self.validation_set is not None:
            self.validation_set.shuffle()
