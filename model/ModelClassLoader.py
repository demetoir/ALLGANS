from util.Logger import StdoutOnlyLogger
from env_settting import *
from util.misc_util import *


class ModelClassLoader:
    def __init__(self):
        self.logger = StdoutOnlyLogger(self.__class__.__name__)
        self.log = self.logger.get_log()

    @staticmethod
    def load_model_class(model_name):
        module_path = module_path_finder(MODEL_MODULE_PATH, model_name)
        model = import_class_from_module_path(module_path, model_name)
        return model
