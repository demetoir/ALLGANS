from visualizer.AbstractVisualizer import AbstractVisualizer
import numpy as np


def get_total_acc(sess, model, next_batch_func, batch_loop, total_size):
    accs = []
    for i in range(batch_loop):
        Xs, Ys = next_batch_func(
            model.batch_size,
            batch_keys=['Xs', 'Ys']
        )
        accs += [model.get_tf_values(sess, model.acc, Xs=Xs, Ys=Ys)]
    acc = np.concatenate(accs, axis=0)[:total_size]
    acc = sum(acc) / total_size
    return acc


class log_titanic_loss(AbstractVisualizer):
    """visualize log of classifier's loss"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        Xs, Ys = dataset.train_set.next_batch(
            model.batch_size,
            batch_keys=['Xs', 'Ys']
        )
        train_loss, global_step = model.get_tf_values(sess, [model.loss_mean, model.global_step], Xs, Ys)

        dataset.validation_set.reset_cursor()
        Xs, Ys = dataset.validation_set.next_batch(
            model.batch_size,
            batch_keys=['Xs', 'Ys']
        )
        valid_loss = model.get_tf_values(sess, model.loss_mean, Xs, Ys)

        batch_loop = dataset.train_set.data_size // model.batch_size + 1
        next_batch = dataset.train_set.next_batch
        train_acc = get_total_acc(sess, model, next_batch, batch_loop, dataset.train_set.data_size)

        batch_loop = dataset.validation_set.data_size // model.batch_size + 1
        next_batch = dataset.validation_set.next_batch
        dataset.validation_set.reset_cursor()
        valid_acc = get_total_acc(sess, model, next_batch, batch_loop, dataset.validation_set.data_size)

        self.log(
            'global_step : %04d ' % global_step,
            'train loss: {:2.4f} '.format(train_loss),
            'valid loss: {:2.4f} '.format(valid_loss),
            'train acc: {:2.4f} '.format(train_acc),
            'valid acc: {:2.4f} '.format(valid_acc),
        )
