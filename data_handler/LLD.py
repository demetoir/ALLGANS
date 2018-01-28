from __future__ import division
from glob import glob
from PIL import Image
from data_handler.AbstractDataset import AbstractDataset
from dict_keys.dataset_batch_keys import *
import _pickle as cPickle
import os
import numpy as np

from util.util import download_data, extract_tar, extract_zip


class LLD(AbstractDataset):
    LLD_CLEAN = 'CLEAN'
    LLD_FULL = 'FULL'
    PATTERN = 'LLD_favicon_data*.pkl'
    SOURCE_URL = "https://data.vision.ee.ethz.ch/cvl/lld_data/LLD_favicons_clean.zip"
    SOURCE_FILE = "LLD_favicons_clean.zip"

    def __init__(self, preprocess=None, batch_after_task=None):
        super().__init__(preprocess, batch_after_task)
        self.batch_keys = [LLD_CLEAN]

    def __repr__(self):
        return 'Large Label Data set'

    def load(self, path, limit=None):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

        download = False
        # TODO hack better file inspection
        files = glob(os.path.join(path, self.PATTERN))
        if len(files) != 5:
            download = True

        if download:
            head, tail = os.path.split(path)
            download_file = os.path.join(head, self.SOURCE_FILE)
            self.log('download %s at %s ' % (self.SOURCE_FILE, download_file))
            download_data(source_url=self.SOURCE_URL,
                          download_path=download_file)

            self.log("extract %s at %s" % (self.SOURCE_FILE, head))
            extract_zip(download_file, head)

        files = glob(os.path.join(path, self.PATTERN))
        files.sort()
        self.data[BATCH_KEY_TRAIN_X] = None
        for file in files:
            with open(file, 'rb') as f:
                data = cPickle.load(f, encoding='latin1')
                self.log('pickle load :%s' % file)

            if self.data[BATCH_KEY_TRAIN_X] is None:
                self.data[BATCH_KEY_TRAIN_X] = data
            else:
                self.data[BATCH_KEY_TRAIN_X] = np.concatenate((self.data[BATCH_KEY_TRAIN_X], data))

        if limit is not None:
            self.data[BATCH_KEY_TRAIN_X] = self.data[BATCH_KEY_TRAIN_X][:limit]

        self.cursor[BATCH_KEY_TRAIN_X] = 0
        self.data_size = len(self.data[BATCH_KEY_TRAIN_X])

        self.log('data set fully loaded')

        super().load(path, limit)

    def save(self):
        # def save_icon_data(icons, data_path, package_size=100000):
        #     if not os.instance_path.exists(data_path):
        #         os.makedirs(data_path)
        #     num_packages = int(math.ceil(len(icons) / package_size))
        #     num_len = len(str(num_packages))
        #     for p in range(num_packages):
        #         with open(os.instance_path.join(data_path, 'icon_data_' + str(p).zfill(num_len) + '.pkl'), 'wb') as f:
        #             cPickle.dump(icons[p * package_size:(p + 1) * package_size], f, protocol=cPickle.HIGHEST_PROTOCOL)
        raise NotImplementedError

    @staticmethod
    def load_sample(path):
        files = glob(os.path.join(path, '5klogos', '*.png'))
        files.sort()

        imgs = []
        for file in files:
            img = Image.open(file)
            img.load_model_instance()
            im_arr = np.fromstring(img.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((img.size[1], img.size[0], 3))
            imgs += [im_arr]

        return np.array(imgs)
