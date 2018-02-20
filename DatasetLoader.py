from env_settting import *
from util.Logger import Logger
from data_handler.CIFAR10 import CIFAR10
from data_handler.CIFAR10 import CIFAR10Helper
from data_handler.CIFAR100 import CIFAR100
from data_handler.CIFAR100 import CIFAR100Helper
from data_handler.LLD import LLD
from data_handler.LLD import LLDHelper
from data_handler.Fashion_MNIST import Fashion_MNIST
from data_handler.Fashion_MNIST import Fashion_MNISTHelper
from data_handler.MNIST import MNIST
from data_handler.MNIST import MNISTHelper


class DatasetLoader:
    """
    Todo
    """

    def __init__(self, root_path=ROOT_PATH):
        """create DatasetManager
        todo
        """
        self.root_path = root_path
        self.logger = Logger(self.__class__.__name__, self.root_path)
        self.log = self.logger.get_log()
        self.datasets = {
            "CIFAR10": (CIFAR10, CIFAR10Helper),
            "CIFAR100": (CIFAR100, CIFAR100Helper),
            "MNIST": (MNIST, MNISTHelper),
            "LLD": (LLD, LLDHelper),
            "fashion-mnist": (Fashion_MNIST, Fashion_MNISTHelper),
        }

    def __repr__(self):
        return self.__class__.__name__

    def load_dataset(self, dataset_name, limit=None):
        """load dataset, return dataset, input_shapes

        :type dataset_name: str
        :type limit: int
        :param dataset_name: dataset name to load
        :param limit: limit dataset_size

        :return: dataset, input_shapes

        :raise KeyError
        invalid dataset_name
        """
        try:
            data_loader, data_helper = self.datasets[dataset_name]
            dataset, input_shapes = data_helper.load_dataset(limit=limit)
        except KeyError:
            raise KeyError("dataset_name %s not found" % dataset_name)

        return dataset, input_shapes
