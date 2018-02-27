from dict_keys.dataset_batch_keys import *
from visualizer.AbstractVisualizer import AbstractVisualizer


class log_D_AE_error(AbstractVisualizer):
    """visualizer error of discriminator AE in BEGAN"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        noise = model.get_noise()
        batch_xs = dataset.next_batch(model.batch_size, batch_keys=[BATCH_KEY_TRAIN_X], lookup=True)
        D_real_AE_error, D_gen_AE_error, global_step = sess.run(
            [model.D_real_AE_error, model.D_gen_AE_error, model.global_step],
            feed_dict={model.z: noise, model.X: batch_xs})

        self.log(
            'global_step : %04d ' % global_step,
            'real AE error: {:.4} '.format(D_real_AE_error),
        )
