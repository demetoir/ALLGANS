from visualizer.AbstractPrintLog import AbstractPrintLog
from dict_keys.dataset_batch_keys import *


class print_D_AE_error(AbstractPrintLog):
    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        noise = model.get_noise()
        batch_xs = dataset.next_batch(model.batch_size, batch_keys=[BATCH_KEY_TRAIN_X], lookup=True)
        D_real_AE_error, D_gen_AE_error, M = sess.run([model.D_real_AE_error, model.D_gen_AE_error, model.M],
                                                      feed_dict={model.z: noise, model.X: batch_xs})
        self.log('real AE error ' + str(D_real_AE_error))
        self.log('gen  AE error ' + str(D_gen_AE_error))
        self.log('Convergence measure ' + str(M))
