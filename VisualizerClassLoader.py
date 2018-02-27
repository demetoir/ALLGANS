from env_settting import *
from util.misc_util import *


class VisualizerClassLoader:
    @staticmethod
    def load_class(model_name):
        module_path = module_path_finder(VISUALIZER_MODULE_PATH, model_name)
        model = import_class_from_module_path(module_path, model_name)
        return model
