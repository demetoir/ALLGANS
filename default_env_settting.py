import os

FILE = 'default_env_setting.py'
ENV_SETTING_PATH = os.path.dirname(os.path.realpath(FILE))
ROOT_PATH = ENV_SETTING_PATH

# model
MODEL_MODULE_PATH = os.path.join(ROOT_PATH, 'model')

# visualizer
VISUALIZER_MODULE_PATH = os.path.join(ROOT_PATH, 'visualizer')

# dataset
DATA_PATH = os.path.join(ROOT_PATH, 'data')
LLD_PATH = os.path.join(DATA_PATH, 'LLD_favicons_clean')
MNIST_PATH = os.path.join(DATA_PATH, 'mnist')
CIFAR10_PATH = os.path.join(DATA_PATH, 'cifar-10-batches-py')
CIFAR100_PATH = os.path.join(DATA_PATH, "cifar-100-python")
FASHION_MNIST_PATH = os.path.join(DATA_PATH, "fashionmnist")
CELEBA_PATH = os.path.join(DATA_PATH, 'img_align_celeba')

# instance
INSTANCE_PATH = os.path.join(ROOT_PATH, 'instance')


def tensorboard_dir():
    import tensorboard
    path, _ = os.path.split(tensorboard.__file__)
    tensorboard_main = os.path.join(path, 'main.py')
    del tensorboard
    return tensorboard_main


templet_env_setting = """
import os

FILE = 'env_setting.py'
ENV_SETTING_PATH = os.path.dirname(os.path.realpath(FILE))
ROOT_PATH = ENV_SETTING_PATH

# model
MODEL_MODULE_PATH = os.path.join(ROOT_PATH, 'model')

# visualizer
VISUALIZER_MODULE_PATH = os.path.join(ROOT_PATH, 'visualizer')

# dataset
DATA_PATH = os.path.join(ROOT_PATH, 'data')
LLD_PATH = os.path.join(DATA_PATH, 'LLD_favicons_clean')
MNIST_PATH = os.path.join(DATA_PATH, 'mnist')
CIFAR10_PATH = os.path.join(DATA_PATH, 'cifar-10-batches-py')
CIFAR100_PATH = os.path.join(DATA_PATH, "cifar-100-python")
FASHION_MNIST_PATH = os.path.join(DATA_PATH, "fashionmnist")
CELEBA_PATH = os.path.join(DATA_PATH, 'img_align_celeba')

# instance
INSTANCE_PATH = os.path.join(ROOT_PATH, 'instance')


def tensorboard_dir():
    import tensorboard
    path, _ = os.path.split(tensorboard.__file__)
    tensorboard_main = os.path.join(path, 'main.py')
    del tensorboard
    return tensorboard_main
"""
