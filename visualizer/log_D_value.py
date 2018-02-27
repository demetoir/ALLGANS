from dict_keys.dataset_batch_keys import *
from visualizer.AbstractVisualizer import AbstractVisualizer


class log_D_value(AbstractVisualizer):
    """visualize log for GAN's Discriminator value"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        noise = model.get_noise()
        batch_xs = dataset.next_batch(model.batch_size, batch_keys=[BATCH_KEY_TRAIN_X], lookup=True)

        D_real, D_gen = sess.run([model.D_real, model.D_gen],
                                 feed_dict={model.z: noise, model.X: batch_xs})
        D_gen = D_gen[:5]
        D_real = D_real[:5]
        self.log(
            'real' + " ".join(map(str, D_real)),
            'gen ' + " ".join(map(str, D_gen))
        )
