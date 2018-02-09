from workbench.InstanceManagerHelper import InstanceManagerHelper
from env_settting import *

# TODO dynamic module load class or function for import instance and visualizer
from visualizer.image_tile import image_tile
from visualizer.log_D_value import log_D_value
from visualizer.log_GAN_loss import log_GAN_loss
from visualizer.image_tile_data import image_tile_data
from visualizer.log_confusion_matrix import log_confusion_matrix
from visualizer.log_classifier_loss import log_classifier_loss

# dataset
from workbench.MNISTHelper import MNISTHelper
from workbench.fashion_MNISTHelper import fashion_MNISTHelper
from workbench.cifar10Helper import cifar10Helper
from workbench.cifar100Helper import cifar100Helper
from workbench.MNISTHelper import MNISTHelper
from workbench.LLDHelper import LLD_helper


# TODO 이거 로드하는부분이랑 경로 찾는거랑 분리하기
def load_model_class_from_module(module_path, class_name):
    from util.misc_util import load_class_from_source_path
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
        print("instance class '%s' not found" % class_name)

    return model


def main():
    dataset, input_shapes = MNISTHelper.load_dataset(limit=1000)
    from unit_test.DummyVisualizer import DummyVisualizer_1
    from unit_test.DummyVisualizer import DummyVisualizer_2
    visualizers = [(DummyVisualizer_1, 40), (DummyVisualizer_2, 40), ]
    from unit_test.DummyModel import DummyModel

    model = DummyModel
    InstanceManagerHelper.build_and_train(model=model,
                                          input_shapes=input_shapes,
                                          visualizers=visualizers,
                                          env_path=ROOT_PATH,
                                          dataset=dataset,
                                          epoch_time=2)
