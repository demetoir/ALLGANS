from visualizer.AbstractVisualizer import AbstractVisualizer


class DummyVisualizer_1(AbstractVisualizer):
    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        print("dummy visualizer1")
        pass


class DummyVisualizer_2(AbstractVisualizer):
    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        print("dummy visualizer2")
        pass
