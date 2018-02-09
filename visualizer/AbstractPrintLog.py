from visualizer.AbstractVisualizer import AbstractVisualizer
from util.Logger import Logger


class AbstractPrintLog(AbstractVisualizer):
    def __init__(self, path=None, execute_interval=None, name=None):
        super().__init__(path, execute_interval, name)
        self.logger = Logger(self.__class__.__name__, self.visualizer_path)
        self.log = self.logger.get_log()

    def __del__(self):
        super().__del__()
        del self.log
        del self.logger

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        raise NotImplementedError
