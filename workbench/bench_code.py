from workbench.InstanceManagerHelper import InstanceManagerHelper
from DatasetLoader import DatasetLoader
from visualizer.image_tile import image_tile
from visualizer.log_GAN_loss import log_GAN_loss
from visualizer.image_tile_data import image_tile_data
from ModelClassLoader import ModelClassLoader


def main():
    dataset, input_shapes = DatasetLoader().load_dataset("MNIST")
    visualizers = [(image_tile, 20), (log_GAN_loss, 10), (image_tile_data, 20)]
    model = ModelClassLoader.load_model_class("GAN")
    InstanceManagerHelper.build_and_train(model=model,
                                          input_shapes=input_shapes,
                                          visualizers=visualizers,
                                          dataset=dataset,
                                          epoch_time=2)

    pass
