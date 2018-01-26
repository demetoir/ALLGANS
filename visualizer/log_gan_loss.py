from visualizer.AbstractPrintLog import AbstractPrintLog
from dict_keys.dataset_batch_keys import *


class print_GAN_loss(AbstractPrintLog):
    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        noise = model.get_noise()
        batch_xs = dataset.next_batch(model.batch_size, batch_keys=[BATCH_KEY_TRAIN_X], lookup=True)
        loss_D, loss_G, global_step = sess.run([model.loss_D, model.loss_G, model.global_step],
                                               feed_dict={model.z: noise, model.X: batch_xs})

        self.log('global_step : %04d ' % global_step
                 + 'D loss: {:.4} '.format(loss_D)
                 + 'G loss: {:.4} '.format(loss_G))
