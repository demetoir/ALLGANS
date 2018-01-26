import os


class AbstractVisualizer:
    def __init__(self, path=None, iter_cycle=None, name=None):
        self.iter_cycle = iter_cycle
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
        del self.iter_cycle
        del self.name
        del self.visualizer_path

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        raise NotImplementedError
