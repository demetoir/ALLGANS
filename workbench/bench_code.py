from InstanceManger import InstanceManager
from workbench.InstanceManagerHelper import InstanceManagerHelper
from DatasetLoader import DatasetLoader
from VisualizerClassLoader import VisualizerClassLoader
from ModelClassLoader import ModelClassLoader

# dataset, input_shapes = DatasetLoader().load_dataset("CIFAR10")
# dataset, input_shapes = DatasetLoader().load_dataset("CIFAR100")
# dataset, input_shapes = DatasetLoader().load_dataset("LLD")
# dataset, input_shapes = DatasetLoader().load_dataset("MNIST")
# dataset, input_shapes = DatasetLoader().load_dataset("Fashion_MNIST")
# model = ModelClassLoader.load_model_class("GAN")

from data_handler.titanic import *
from sklearn import tree


class DecisionTreeClassifier:
    """
    sklearn base DecisionTreeClassifier
    """

    def __init__(self, *args, **kwargs):
        self.d_tree = tree.DecisionTreeClassifier(*args, **kwargs)

    def fit(self, xs, labels):
        self.d_tree.fit(xs, labels)

    def predicts(self, xs):
        return self.d_tree.predict(xs)

    def acc(self, xs, labels):
        return self.d_tree.score(xs, labels)

    def probs(self, xs, ):
        """
        if multi label than output shape == (class, sample, prob)
        need to transpose shape to (sample, class, prob)

        :param xs:
        :return:
        """
        probs = np.array(self.d_tree.predict_proba(xs))
        probs = np.transpose(probs, axes=(1, 0, 2))
        return probs


def sklearn_DecisionTreeClassifier(dataset, max_depth):
    dtree = DecisionTreeClassifier(max_depth=max_depth)
    print(id(dtree))
    batch_xs, batch_labels = dataset.train_set.next_batch(
        dataset.train_set.data_size,
        batch_keys=[BK_X, BK_LABEL]
    )

    dtree.fit(batch_xs, batch_labels)

    acc = dtree.acc(batch_xs, batch_labels)
    print("train acc:", acc)

    batch_xs, batch_labels = dataset.validation_set.next_batch(
        dataset.validation_set.data_size,
        batch_keys=[BK_X, BK_LABEL]
    )
    acc = dtree.acc(batch_xs, batch_labels)
    print("valid acc:", acc)


def main():
    dataset = DatasetLoader().load_dataset("titanic")
    input_shapes = dataset.train_set.input_shapes
    for i in range(1, 10):
        dataset.train_set.shuffle()
        sklearn_DecisionTreeClassifier(dataset, max_depth=i)

    # model = ModelClassLoader.load_model_class("TitanicModel")
    #
    # manager = InstanceManager()
    # metadata_path = manager.build_instance(model)
    # manager.load_instance(metadata_path, input_shapes)
    #
    # visualizers = [(log_titanic_loss, 25), (log_confusion_matrix, 500)]
    # for visualizer, interval in visualizers:
    #     manager.load_visualizer(visualizer, interval)
    #
    # manager.train_instance(
    #     epoch=4000,
    #     dataset=dataset,
    #     check_point_interval=5000,
    #     with_tensorboard=False)
    # manager.sampling_instance(
    #     dataset=dataset
    # )
    #
    # del manager
