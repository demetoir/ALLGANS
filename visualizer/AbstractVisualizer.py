import os


class AbstractVisualizer:
    """

    """
    def __init__(self, path=None, execute_interval=None, name=None):
        """create AbstractVisualizer

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

    def __str__(self):
        if self.name is not None:
            return self.name
        else:
            return self.__class__.__name__

    def __del__(self):
        del self.execute_interval
        del self.name
        del self.visualizer_path

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
