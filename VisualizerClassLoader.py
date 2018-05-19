from env_settting import *
from util.misc_util import *


class VisualizerClassLoader:
    @staticmethod
    def load(class_name, path=VISUALIZER_MODULE_PATH):
        module_path = module_path_finder(path, class_name)
        class_ = import_class_from_module_path(module_path, class_name)
        return class_
