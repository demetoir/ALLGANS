from visualizer.AbstractPrintLog import AbstractPrintLog
from unit_test.DummyModel import DummyModel
from InstanceManger import InstanceManager
from unit_test.DummyVisualizer import DummyVisualizer_1
from unit_test.DummyVisualizer import DummyVisualizer_2
from unit_test.DummyDataset import DummyDataset
from env_settting import *


class dummy_log(AbstractPrintLog):
    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        super().task(sess, iter_num, model, dataset)
        self.log('this is dummy log')


class test_InstanceManger:
    def __init__(self):
        pass

    def test_00_init(self):
        path = ROOT_PATH
        InstanceManager(path)

    def test_01_gen_model(self):
        path = ROOT_PATH
        manager = InstanceManager(path)

        model = DummyModel
        manager.build_instance(model)

    def test_02_load_visualizer(self):
        path = ROOT_PATH
        manager = InstanceManager(path)

        model = DummyModel
        manager.build_instance(model)

        visualizer_1 = DummyVisualizer_1
        visualizer_2 = DummyVisualizer_2

        visualizers = [visualizer_1, visualizer_2]
        manager.load_visualizer(visualizers)

    def test_03_train_model(self):
        path = ROOT_PATH
        manager = InstanceManager(path)

        model = DummyModel
        manager.build_instance(model)

        visualizer_1 = DummyVisualizer_1
        visualizer_2 = DummyVisualizer_2

        visualizers = [visualizer_1, visualizer_2]
        manager.load_visualizer(visualizers)

        dataset = DummyDataset()
        epoch_time = 10
        manager.train_instance(dataset, epoch_time)

    def test_04_save_model(self):
        path = ROOT_PATH
        manager = InstanceManager(path)

        model = DummyModel
        manager.build_instance(model)

        visualizer_1 = DummyVisualizer_1
        visualizer_2 = DummyVisualizer_2

        visualizers = [visualizer_1, visualizer_2]
        manager.load_visualizer(visualizers)

        dataset = DummyDataset()
        epoch_time = 10
        check_point_interval = 2
        manager.train_instance(dataset, epoch_time, check_point_interval)

    def test_05_load_model(self):
        path = ROOT_PATH
        manager = InstanceManager(path)

        model = DummyModel
        input_shape = (32, 32, 3)
        manager.build_instance(model)

        visualizer_1 = DummyVisualizer_1
        visualizer_2 = DummyVisualizer_2

        visualizers = [visualizer_1, visualizer_2]
        manager.load_visualizer(visualizers)

        dataset = DummyDataset()
        epoch_time = 10
        check_point_interval = 2
        manager.train_instance(dataset, epoch_time, check_point_interval)

        #
        manager = InstanceManager(path)
        manager.load_instance(input_shape)

        visualizer_1 = DummyVisualizer_1
        visualizer_2 = DummyVisualizer_2

        visualizers = [visualizer_1, visualizer_2]
        manager.load_visualizer(visualizers)

        dataset = DummyDataset()
        epoch_time = 10
        check_point_interval = 2
        manager.train_instance(dataset, epoch_time, check_point_interval)


if __name__ == '__main__':
    path = ROOT_PATH
    manager = InstanceManager(path)

    model = DummyModel
    input_shape = (32, 32, 3)
    manager.build_instance(model)

    visualizers = [(dummy_log, 10), ]
    manager.load_visualizer(visualizers)

    dataset = DummyDataset()
    epoch_time = 1
    check_point_interval = 500
    manager.train_instance(dataset, epoch_time, check_point_interval)
