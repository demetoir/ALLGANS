from visualizer.AbstractVisualizer import AbstractVisualizer
from util.Logger import Logger


class AbstractPrintLog(AbstractVisualizer):
    """abstract class for visualizing by printing log

    logging object will log on log file and stdout
    """
    def __init__(self, path=None, execute_interval=None, name=None):
        """create print log visualizer"""
        super().__init__(path, execute_interval, name)
        self.logger = Logger(self.__class__.__name__, self.visualizer_path)
        self.log = self.logger.get_log()

    def __del__(self):
        super().__del__()
        del self.log
        del self.logger

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        """visualizing task for print log

        ex)
        def task(self, sess=None, iter_num=None, model=None, dataset=None):
            self.log("log message")

        :type sess: tensorflow.Session
        :type iter_num: int
        :type model: AbstractModel
        :type dataset: AbstractDataset
        :param sess: current tensorflow session
        :param iter_num: current iteration number
        :param model: current visualizing model
        :param dataset: current visualizing dataset

        :raise NotImplementedError
        if not implemented
        """
        raise NotImplementedError
