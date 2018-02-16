from workbench.InstanceManagerHelper import InstanceManagerHelper
from DatasetManager import DatasetManager

from visualizer.image_tile import image_tile
from visualizer.log_GAN_loss import log_GAN_loss
from visualizer.image_tile_data import image_tile_data
from ModelLoader import ModelLoader
from util.misc_util import *
from env_settting import *


def main():
    # path = "C:/Users/demetoir_desktop/PycharmProjects/ALLGANS/model/GAN/GAN.py"
    # name = "GANd"
    # class_ = import_class_from_module_path(path, name)

    dataset, input_shapes = DatasetManager().load_dataset("MNIST")
    visualizers = [(image_tile, 20), (log_GAN_loss, 10), (image_tile_data, 20)]
    model = ModelLoader.load("GAN")
    InstanceManagerHelper.build_and_train(model=model,
                                          input_shapes=input_shapes,
                                          visualizers=visualizers,
                                          dataset=dataset,
                                          epoch_time=2)

    pass
