from InstanceManger import InstanceManager
from workbench.InstanceManagerHelper import InstanceManagerHelper
from DatasetLoader import DatasetLoader
from visualizer.image_tile import image_tile
from visualizer.log_GAN_loss import log_GAN_loss
from visualizer.image_tile_data import image_tile_data
from visualizer.log_classifier_loss import log_classifier_loss

from ModelClassLoader import ModelClassLoader

# dataset, input_shapes = DatasetLoader().load_dataset("CIFAR10")
# dataset, input_shapes = DatasetLoader().load_dataset("CIFAR100")
# dataset, input_shapes = DatasetLoader().load_dataset("LLD")
# dataset, input_shapes = DatasetLoader().load_dataset("MNIST")
# dataset, input_shapes = DatasetLoader().load_dataset("Fashion_MNIST")
# model = ModelClassLoader.load_model_class("GAN")

from data_handler.titanic import *


def main():
    dataset = DatasetLoader().load_dataset("titanic")
    input_shapes = dataset.input_shapes
    exit()

    visualizers = [(log_classifier_loss, 100)]
    model = ModelClassLoader.load_model_class("TitanicModel")

    manager = InstanceManager()
    metadata_path = manager.build_instance(model)
    manager.load_instance(metadata_path, input_shapes)
    for visualizer, interval in visualizers:
        manager.load_visualizer(visualizer, interval)

    manager.train_instance(
        epoch=2000,
        dataset=dataset,
        check_point_interval=5000)

    manager.sampling_instance()

    del manager
