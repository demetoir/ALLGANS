from util.Logger import Logger
from env_settting import *
from util.misc_util import *


class ModelLoader:
    def __init__(self):
        self.logger = Logger(self.__class__.__name__, stdout_only=True)
        self.log = self.logger.get_log()

    @staticmethod
    def load(model_name):
        module_path = module_path_finder(MODEL_MODULE_PATH, model_name)
        model = import_class_from_module_path(module_path, model_name)
        return model
