from visualizer.AbstractVisualizer import AbstractVisualizer


class DummyVisualizer_1(AbstractVisualizer):
    def __init__(self, path, name=None):
        super().__init__(path, name)
        self.iter_cycle = 5
        self.name = 'dummy 1'

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        print(self.__class__.__name__)


class DummyVisualizer_2(AbstractVisualizer):
    def __init__(self, path, name=None):
        super().__init__(path, name)
        self.iter_cycle = 5

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        print(self.__class__.__name__)
