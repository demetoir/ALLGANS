from DatasetManager import DatasetManager
from InstanceManger import InstanceManager
from ModelLoader import ModelLoader
from visualizer.image_tile import image_tile
from visualizer.image_tile_data import image_tile_data
from visualizer.log_GAN_loss import log_GAN_loss


dataset, input_shapes = DatasetManager().load_dataset("CIFAR10")

model = ModelLoader.load("GAN")

manager = InstanceManager()
metadata_path = manager.build_instance(model)

manager.load_instance(metadata_path, input_shapes)

manager.load_visualizer(image_tile, 20)
manager.load_visualizer(log_GAN_loss, 10)
manager.load_visualizer(image_tile_data, 20)
manager.train_instance(epoch=10, dataset=dataset)
