from __future__ import division
from glob import glob
from PIL import Image
from data_handler.AbstractDataset import AbstractDataset, DownloadInfo
from dict_keys.dataset_batch_keys import *
from env_settting import LLD_PATH
from dict_keys.input_shape_keys import *
import _pickle as cPickle
import os
import numpy as np


class LLD(AbstractDataset):
    LLD_CLEAN = 'CLEAN'
    LLD_FULL = 'FULL'
    PATTERN = 'LLD_favicon_data*.pkl'

    def __init__(self, preprocess=None, batch_after_task=None):
        super().__init__(preprocess, batch_after_task)
        self.batch_keys = [
            LLD_CLEAN,
        ]

        self.download_infos = [
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

            if self.data[BATCH_KEY_TRAIN_X] is None:
                self.data[BATCH_KEY_TRAIN_X] = data
            else:
                self.data[BATCH_KEY_TRAIN_X] = np.concatenate((self.data[BATCH_KEY_TRAIN_X], data))

        if limit is not None:
            self.data[BATCH_KEY_TRAIN_X] = self.data[BATCH_KEY_TRAIN_X][:limit]

        self.cursor[BATCH_KEY_TRAIN_X] = 0
        self.data_size = len(self.data[BATCH_KEY_TRAIN_X])

        self.log('data set fully loaded')

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


class LLDHelper:
    @staticmethod
    def next_batch_task(batch):
        x = batch[0]
        return x

    @staticmethod
    def load_dataset(limit=None):
        lld_data = LLD(batch_after_task=LLDHelper.next_batch_task)
        lld_data.load(LLD_PATH, limit=limit)
        input_shapes = {
            INPUT_SHAPE_KEY_DATA_X: [32, 32, 3],
        }

        return lld_data, input_shapes
