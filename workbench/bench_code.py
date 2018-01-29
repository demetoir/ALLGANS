from workbench.InstanceManagerHelper import InstanceManagerHelper
from env_settting import *

# TODO dynamic module load class or function for import model and visualizer
from visualizer.image_tile import image_tile
from visualizer.log_D_value import print_D_value
from visualizer.log_gan_loss import print_GAN_loss
from visualizer.image_tile_data import image_tile_data
from visualizer.log_confusion_matrix import print_confusion_matrix
from visualizer.log_classifier_loss import print_classifier_loss

# dataset
from workbench.MNISTHelper import MNISTHelper
from workbench.fashion_MNISTHelper import fashion_MNISTHelper
from workbench.cifar10Helper import cifar10Helper
from workbench.MNISTHelper import MNISTHelper
from workbench.LLDHelper import LLD_helper


# TODO NHWC format is right not NWHC
# NWHC = Num_samples x Height x Width x Channels

def load_model_class_from_module(module_path, class_name):
    from util.util import load_class_from_source_path
    from glob import glob

    model = None
    paths = glob(os.path.join(module_path, '**', '*.py'), recursive=True)
    for path in paths:
        file_name = os.path.basename(path)
        if file_name == class_name + '.py':
            model = load_class_from_source_path(path, class_name)
            print('load class %s from %s' % (model, path))
    del load_class_from_source_path
    del glob
    if model is None:
        print("model class '%s' not found" % class_name)

    return model


def main():
    dataset, input_shapes = cifar10Helper.load_dataset(limit=1000)

    # dataset, input_shapes = MNISTHelper.load_dataset(limit=1000)
    # visualizers = [(image_tile, 40), (image_tile_data, 100), (print_GAN_loss, 10), (print_D_value, 10)]
    # model = load_model_class_from_module(MODEL_MODULE_PATH, 'GAN')
    # InstanceManagerHelper.gen_model_and_train(model=model,
    #                                           input_shapes=input_shapes,
    #                                           visualizers=visualizers,
    #                                           env_path=ROOT_PATH,
    #                                           dataset=dataset,
    #                                           epoch_time=5)
    #

