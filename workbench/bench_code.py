from InstanceManger import InstanceManager
from workbench.InstanceManagerHelper import InstanceManagerHelper
from DatasetLoader import DatasetLoader
from visualizer.image_tile import image_tile
from visualizer.log_GAN_loss import log_GAN_loss
from visualizer.image_tile_data import image_tile_data
from visualizer.log_classifier_loss import log_classifier_loss
from visualizer.log_confusion_matrix import log_confusion_matrix
from visualizer.log_titanic_loss import log_titanic_loss
from visualizer.csv_titanic_result import csv_titanic_result
from ModelClassLoader import ModelClassLoader
from sklearn import tree

# dataset, input_shapes = DatasetLoader().load_dataset("CIFAR10")
# dataset, input_shapes = DatasetLoader().load_dataset("CIFAR100")
# dataset, input_shapes = DatasetLoader().load_dataset("LLD")
# dataset, input_shapes = DatasetLoader().load_dataset("MNIST")
# dataset, input_shapes = DatasetLoader().load_dataset("Fashion_MNIST")
# model = ModelClassLoader.load_model_class("GAN")

from data_handler.titanic import *
import guitarpro


def sklearn_DecisionTreeClassifier(dataset):
    print("sklearn Decision Tree Classifier")

    dtree = tree.DecisionTreeClassifier()
    batch_xs, batch_labels = dataset.train_set.next_batch(
        dataset.train_set.data_size - 1,
        batch_keys=[BK_X, BK_LABEL]
    )
    dtree = dtree.fit(batch_xs, batch_labels)

    acc = 0.0
    res = dtree.predict(batch_xs)
    for i in range(len(batch_labels)):
        if np.array_equal(batch_labels[i], res[i]):
            acc += 1
    acc /= len(batch_labels)
    print("train acc:", acc)

    batch_xs, batch_labels = dataset.validation_set.next_batch(
        dataset.validation_set.data_size - 1,
        batch_keys=[BK_X, BK_LABEL]
    )
    acc = 0.0
    res = dtree.predict(batch_xs)
    for i in range(len(batch_labels)):
        if np.array_equal(batch_labels[i], res[i]):
            acc += 1
    acc /= len(batch_labels)
    print("valid acc:", acc)
    print()


def main():
    dataset = DatasetLoader().load_dataset("titanic")
    input_shapes = dataset.train_set.input_shapes
    sklearn_DecisionTreeClassifier(dataset)

    model = ModelClassLoader.load_model_class("TitanicModel")

    manager = InstanceManager()
    metadata_path = manager.build_instance(model)
    manager.load_instance(metadata_path, input_shapes)

    visualizers = [(log_titanic_loss, 25), (log_confusion_matrix, 500)]
    for visualizer, interval in visualizers:
        manager.load_visualizer(visualizer, interval)

    manager.train_instance(
        epoch=4000,
        dataset=dataset,
        check_point_interval=5000,
        with_tensorboard=False)
    manager.sampling_instance(
        dataset=dataset
    )

    del manager
