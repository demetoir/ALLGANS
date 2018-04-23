from pprint import pprint

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

from workbench.sklearn_toolkit import *


def fit_and_test(model, dataset):
    params = {}
    instance = model(**params)
    print(instance)
    print("params")
    params = instance.get_params()
    pprint(params)

    batch_xs, batch_labels = dataset.train_set.next_batch(
        dataset.train_set.data_size,
        batch_keys=[BK_X, BK_LABEL],
    )
    instance.fit(batch_xs, batch_labels, Ys_type="onehot")

    acc = instance.score(batch_xs, batch_labels, Ys_type="onehot")
    print("train acc:", acc)

    batch_xs, batch_labels = dataset.validation_set.next_batch(
        dataset.validation_set.data_size,
        batch_keys=[BK_X, BK_LABEL]
    )
    acc = instance.score(batch_xs, batch_labels, Ys_type="onehot")
    print("valid acc:", acc)

    print("probs")
    probs = instance.proba(batch_xs[:10], transpose_shape=False)
    print(probs)

    print("predict")
    print(instance.predict(batch_xs[:10]))

    print()


def param_search(instance, dataset):
    param_grid = {
        'alpha': [1.0, 2.0, 4.0],
        'binarize': [0.0, 0.5, 1.0, 2.0],
        'class_prior': [None],
        'fit_prior': [True, False],
    }

    gs = ParamGridSearch(instance, param_grid)
    batch_xs, batch_labels = dataset.train_set.next_batch(
        dataset.train_set.data_size,
        batch_keys=[BK_X, BK_LABEL],
    )
    gs.fit(batch_xs, batch_labels)

    d = gs.cv_results_
    # pprint(d)

    print("best")
    print(gs.best_score_)
    print(gs.best_params_)

    instance.set_params(**gs.best_params_)
    batch_xs, batch_labels = dataset.train_set.next_batch(
        dataset.train_set.data_size,
        batch_keys=[BK_X, BK_LABEL],
    )
    instance.fit(batch_xs, batch_labels, Ys_type="onehot")

    acc = instance.score(batch_xs, batch_labels, Ys_type="onehot")
    print("train acc:", acc)

    batch_xs, batch_labels = dataset.validation_set.next_batch(
        dataset.validation_set.data_size,
        batch_keys=[BK_X, BK_LABEL]
    )
    acc = instance.score(batch_xs, batch_labels, Ys_type="onehot")
    print("valid acc:", acc)


classifiers = [
    MLP,

    SGD,
    Gaussian_NB,
    Bernoulli_NB,
    Multinomial_NB,
    DecisionTree,
    RandomForest,
    ExtraTrees,
    AdaBoost,
    GradientBoosting,
    QDA,
    KNeighbors,
    Linear_SVC,
    RBF_SVM,
    GaussianProcess,
]

import os



def main():
    import xgboost as xgb
    pprint(xgb)


    # dataset = DatasetLoader().load_dataset("titanic")
    # input_shapes = dataset.train_set.input_shapes
    #
    # for clf in classifiers:
    #     fit_and_test(clf, dataset)

    # model = ModelClassLoader.load_model_class("TitanicModel")
    #
    # manager = InstanceManager()
    # metadata_path = manager.build_instance(model)
    # manager.load_instance(metadata_path, input_shapes)
    #
    # log_titanic_loss = VisualizerClassLoader.load('log_titanic_loss')
    # log_confusion_matrix = VisualizerClassLoader.load('log_confusion_matrix')
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
