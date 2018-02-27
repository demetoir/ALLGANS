from data_handler.AbstractDataset import AbstractDatasetHelper
from data_handler.AbstractDataset import AbstractDataset
from env_settting import *
from util.Logger import Logger
from util.misc_util import *


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
        self.datasets = {}

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
            if dataset_name not in self.datasets:
                self.import_dataset_and_helper(dataset_name=dataset_name)
            data_loader, data_helper = self.datasets[dataset_name]
            path = os.path.join(DATA_PATH, data_loader.__name__)
            dataset, input_shapes = data_helper.load_dataset(path=path, limit=limit)
        except KeyError:
            raise KeyError("dataset_name %s not found" % dataset_name)

        return dataset, input_shapes

    def import_dataset_and_helper(self, dataset_name):
        """ import dataset_and_helper

        :type dataset_name: str
        :param dataset_name:
        """
        self.log('load %s dataset module' % dataset_name)
        paths = glob(os.path.join(DATA_HANDLER_PATH, '**', '*.py'), recursive=True)

        dataset_path = None
        for path in paths:
            _, file_name = os.path.split(path)
            dataset_name_ = file_name.replace('.py', '')
            if dataset_name != dataset_name_:
                continue
            dataset_path = path

        if dataset_path is None:
            raise ModuleNotFoundError("dataset %s not found" % dataset_name)

        module_ = import_module_from_module_path(dataset_path)
        dataset = None
        helper = None
        for key in module_.__dict__:
            value = module_.__dict__[key]
            try:
                if issubclass(value, AbstractDataset):
                    dataset = value
                if issubclass(value, AbstractDatasetHelper):
                    helper = value
            except TypeError:
                pass

        if dataset is None:
            raise ModuleNotFoundError("dataset class %s not found" % dataset_name)
        if helper is None:
            raise ModuleNotFoundError("dataset helper class %s not found" % dataset_name)

        self.datasets[dataset_name] = (dataset, helper)
