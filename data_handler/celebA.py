from glob import glob
from data_handler.AbstractDataset import AbstractDataset

import numpy as np
import time
import os
import pickle
import cv2


class celebA(AbstractDataset):
    PATTERN_pkl_img = "celebA_img_*.pkl"
    PATTERN_pkl_label = "celebA_label_*.pkl"
    PATTERN_source_img = "*.jpg"
    PATTERN_source_label = None
    PATTHERN_pkl_folder = "celebA_pkl"

    DATA_x = "data_x"
    DATA_label = "data_label"

    def check_build_pkl(self, path):
        files = glob(os.path.join(path, self.PATTERN_pkl_img))
        if len(files) == 0:
            return False

        files = glob(os.path.join(path, self.PATTERN_pkl_label))
        if len(files) == 0:
            return False
        return True

    def build_pkl(self, source_path):
        head, tail = os.path.split(source_path)
        pkl_path = os.path.join(head, self.PATTHERN_pkl_folder)
        print(pkl_path)
        "/mnt/44837BD87A41C801/GAN-root/dataset/celebA_pkl"

        # build img pkl
        files = glob(os.path.join(source_path, self.PATTERN_source_img))
        files.sort()
        start = time.time()
        data = []
        bucket_size = 10000
        # files = files[100000:]
        for i, file in enumerate(files):
            np_img = cv2.imread(file)
            # print(np_img.shape)
            np_img = np.reshape(np_img, (1, *np_img.shape))
            data += [np_img]

            if i % 1000 == 0:
                print(time.time() - start)
                print(len(data))
                print("load %d / %d done" % (i, len(files)))

            if (i + 1) % bucket_size == 0:
                data = np.concatenate(data)
                save_file = "celebA_img_%d.pkl" % ((i + 1) // bucket_size)
                with open(os.path.join(pkl_path, save_file), mode='wb') as f:
                    pickle.dump(data, f)
                data = []

                print("dump %s done" % save_file)

    def __init__(self, preprocess=None, batch_after_task=None):
        super().__init__(preprocess, batch_after_task)
        # TODO add keys
        self.keys = None

    @staticmethod
    def __load_data(files):
        data = None
        for file in files:
            with open(file, mode="wb") as f:
                obj = pickle.load(f)

            if data is None:
                data = obj
            else:
                data = np.concatenate((data, np))
        return data

    def load(self, path, limit=None):
        # TODO implement here

        # load img
        files = glob(os.path.join(path, self.PATTERN_pkl_img))
        if len(files) == 0:
            raise ValueError("celebA pkl img not found")
        files.sort()
        print(files)
        self.data[self.DATA_x] = self.__load_data(files)

        # load label
        # files = glob(os.path.join(path, self.PATTERN_pkl_label))
        # if len(files) == 0:
        #     raise ValueError("celebA pkl label not found")
        # files.sort()
        # print(files)
        # self.data[self.DATA_label] = self.__load_data(files)

        super().load(path, limit)

    def save(self):
        raise NotImplementedError
