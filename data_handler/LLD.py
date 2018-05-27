from __future__ import division
from glob import glob
from data_handler.BaseDataset import BaseDataset, DownloadInfo, DatasetCollection
from dict_keys.dataset_batch_keys import *
import _pickle as cPickle
import os


class LLD_clean(BaseDataset):
    LLD_CLEAN = 'CLEAN'
    LLD_FULL = 'FULL'
    PATTERN = 'LLD_favicon_data*.pkl'

    @property
    def downloadInfos(self):
        return [
            DownloadInfo(
                url="https://data.vision.ee.ethz.ch/cvl/lld_data/LLD_favicons_clean.zip",
                is_zipped=True,
                download_file_name="LLD_favicons_clean.zip",
                extracted_file_names=[
                    "LLD_favicon_data_0.pkl",
                    "LLD_favicon_data_1.pkl",
                    "LLD_favicon_data_2.pkl",
                    "LLD_favicon_data_3.pkl",
                    "LLD_favicon_data_4.pkl"
                ]
            )
        ]

    def load(self, path, limit=None):
        files = glob(os.path.join(path, self.PATTERN))
        files.sort()
        self.data[BATCH_KEY_TRAIN_X] = None
        for file in files:
            with open(file, 'rb') as f:
                data = cPickle.load(f, encoding='latin1')
                self.log('pickle load :%s' % file)
            self._append_data('Xs', data)

    def save(self):
        # def save_icon_data(icons, data_path, package_size=100000):
        #     if not os.instance_path.exists(data_path):
        #         os.makedirs(data_path)
        #     num_packages = int(math.ceil(len(icons) / package_size))
        #     num_len = len(str(num_packages))
        #     for p in range(num_packages):
        #         with open(os.instance_path.join(data_path, 'icon_data_' + str(p).zfill(num_len) + '.pkl'), 'wb') as f:
        #             cPickle.dump(icons[p * package_size:(p + 1) * package_size], f, protocol=cPickle.HIGHEST_PROTOCOL)
        pass

    def preprocess(self):
        pass


class LLD(DatasetCollection):

    def __init__(self, train_set=None, test_set=None, validation_set=None):
        super().__init__(train_set, test_set, validation_set)
        self.train_set = LLD_clean()
