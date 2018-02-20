import os
from glob import glob

from util.Logger import Logger
from util.numpy_utils import np_img_to_PIL_img


class AbstractVisualizer:
    """abstract class for visualizer for instance

    """

    def __init__(self, path=None, execute_interval=None, name=None):
        """create Visualizer

        :type path: str
        :type execute_interval: int
        :type name: str
        :param path: path for saving visualized result
        :param execute_interval: interval for execute
        :param name: naming for visualizer
        """

        self.execute_interval = execute_interval
        self.name = name
        self.visualizer_path = os.path.join(path, self.__str__())

        if not os.path.exists(path):
            os.mkdir(path)

        if not os.path.exists(self.visualizer_path):
            os.mkdir(self.visualizer_path)

        files = glob(os.path.join(self.visualizer_path, '*'))
        self.output_count = len(files)

        self.logger = Logger(self.__class__.__name__, self.visualizer_path)
        self.log = self.logger.get_log()

    def __str__(self):
        if self.name is not None:
            return self.name
        else:
            return self.__class__.__name__

    def __del__(self):
        del self.execute_interval
        del self.name
        del self.visualizer_path
        del self.log
        del self.logger

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        """visualizing task

        :type sess: tensorflow.Session
        :type iter_num: int
        :type model: AbstractModel
        :type dataset: AbstractDataset
        :param sess: current tensorflow session
        :param iter_num: current iteration number
        :param model: current visualizing model
        :param dataset: current visualizing dataset
        """
        raise NotImplementedError

    def save_np_img(self, np_img, file_name=None):
        """save np_img file in visualizer path

        :type np_img: numpy.Array
        :type file_name: strs
        :param np_img: np_img to save
        :param file_name: save file name
        default None
        if file_name is None, file name of np_img will be 'output_count.png'
        """
        if file_name is None:
            file_name = '{}.png'.format(str(self.output_count).zfill(8))

        pil_img = np_img_to_PIL_img(np_img)
        with open(os.path.join(self.visualizer_path, file_name), 'wb') as fp:
            pil_img.save(fp)
