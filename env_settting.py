import os

FILE = 'env_setting.py'
ROOT_PATH = os.path.dirname(os.path.realpath(FILE))

# model
MODEL_MODULE_PATH = os.path.join(ROOT_PATH, 'model')

# visualizer
VISUALIZER_MODULE_PATH = os.path.join(ROOT_PATH, 'visualizer')

# dataset
DATA_PATH = 'data'
LLD_PATH = os.path.join(ROOT_PATH, DATA_PATH, 'LLD_favicons_clean')
MNIST_PATH = os.path.join(ROOT_PATH, DATA_PATH, 'mnist')
CIFAR10_PATH = os.path.join(ROOT_PATH, DATA_PATH, 'cifar-10-batches-py')
CIFAR100_PATH = os.path.join(ROOT_PATH, DATA_PATH, "cifar-100-python")
FASHION_MNIST_PATH = os.path.join(ROOT_PATH, DATA_PATH, "fashionmnist")
CELEBA_PATH = os.path.join(ROOT_PATH, DATA_PATH, 'img_align_celeba')


def tensorboard_dir():
    import tensorboard
    path, _ = os.path.split(tensorboard.__file__)
    tensorboard_main = os.path.join(path, 'main.py')
    del tensorboard
    return tensorboard_main

