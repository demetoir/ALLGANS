from data_handler.LLD import LLD
from env_settting import ROOT_PATH
from model.GAN import GAN
from InstanceManger import InstanceManager
from visualizer.AbstractPrintLog import AbstractPrintLog

import os


class dummy_log(AbstractPrintLog):
    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        super().task(sess, iter_num, model, dataset)
        self.log('this is dummy log')


if __name__ == '__main__':
    root_path = ROOT_PATH
    LLD_PATH = os.path.join(root_path, 'dataset', 'LLD')
    lld_data = LLD()
    lld_data.load(LLD_PATH)

    manager = InstanceManager(ROOT_PATH)

    input_shape = [32, 32, 3]
    model = GAN
    manager.gen_instance(model, input_shape)
    metadata_path = manager.metadata_path

    iter_cycle = 10
    visualizers = [(dummy_log, iter_cycle)]
    manager.load_visualizer(visualizers)

    epoch_time = 10
    check_point_interval_per_iter = 5000
    manager.train_model(lld_data, epoch_time, check_point_interval_per_iter)
