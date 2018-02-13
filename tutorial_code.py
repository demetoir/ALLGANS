from workbench.InstanceManagerHelper import InstanceManagerHelper
from DatasetManager import DatasetManager


def main():
    from unit_test.DummyVisualizer import DummyVisualizer_1
    from unit_test.DummyVisualizer import DummyVisualizer_2
    from unit_test.DummyModel import DummyModel

    dataset, input_shapes = DatasetManager().load_dataset("MNIST")
    visualizers = [(DummyVisualizer_1, 40), (DummyVisualizer_2, 40), ]
    model = DummyModel
    InstanceManagerHelper.build_and_train(model=model,
                                          input_shapes=input_shapes,
                                          visualizers=visualizers,
                                          dataset=dataset,
                                          epoch_time=2)
