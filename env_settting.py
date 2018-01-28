import os

FILE = 'env_setting.py'
ROOT_PATH = os.path.dirname(os.path.realpath(FILE))

# model
MODEL_MODULE_PATH = os.path.join(ROOT_PATH, 'model')

# visualizer
VISUALIZER_MODULE_PATH = os.path.join(ROOT_PATH, 'visualizer')

# dataset
DATA_PATH = 'data'
LLD_PATH = os.path.join(ROOT_PATH, DATA_PATH, 'LLD')
MNIST_PATH = os.path.join(ROOT_PATH, DATA_PATH, 'mnist')
CIFAR10_PATH = os.path.join(ROOT_PATH, DATA_PATH, 'cifar-10-batches-py')
CIFAR100_PATH = os.path.join(ROOT_PATH, DATA_PATH, "cifar-100-python")
FASHION_MNIST_PATH = os.path.join(ROOT_PATH, DATA_PATH, "fashionmnist")
CELEBA_PATH = os.path.join(ROOT_PATH, DATA_PATH, 'img_align_celeba')


def board_path():
    # TODO : implement find tensorboard path

    import pip
    modules = pip.get_installed_distributions()
    tensorboard = None
    for module_ in modules:
        if "tensorflow-tensorboard" in module_.key:
            tensorboard = module_
            break
    del pip
    print(tensorboard)

    # return 'here past your tensorboard executable file path'
    return '/home/demetoir/anaconda3/envs/tensor/bin/tensorboard'
