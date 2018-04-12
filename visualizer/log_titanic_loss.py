from dict_keys.dataset_batch_keys import *
from visualizer.AbstractVisualizer import AbstractVisualizer
import numpy as np


def get_total_acc(sess, model, next_batch_func, batch_loop, total_size):
    acc_list = []
    for i in range(batch_loop):
        batch_xs, batch_labels = next_batch_func(
            model.batch_size,
            batch_keys=[BK_X, BK_LABEL]
        )
        acc_list += [sess.run(
            model.batch_acc,
            feed_dict={
                model.X: batch_xs,
                model.label: batch_labels,
                model.dropout_rate: 1
            }
        )]
    acc = np.concatenate(acc_list, axis=0)[:total_size]
    acc = sum(acc) / total_size
    return acc


class log_titanic_loss(AbstractVisualizer):
    """visualize log of classifier's loss"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        batch_xs, batch_labels = dataset.train_set.next_batch(
            model.batch_size,
            batch_keys=[BK_X, BK_LABEL]
        )
        loss, global_step = sess.run(
            [model.loss_mean, model.global_step],
            feed_dict={
                model.X: batch_xs,
                model.label: batch_labels,
                model.dropout_rate: 1
            }
        )

        batch_loop = dataset.train_set.data_size // model.batch_size + 1
        next_batch = dataset.train_set.next_batch
        train_acc = get_total_acc(sess, model, next_batch, batch_loop, dataset.train_set.data_size)

        batch_loop = dataset.validation_set.data_size // model.batch_size + 1
        next_batch = dataset.validation_set.next_batch
        dataset.validation_set.reset_cursor()
        valid_acc = get_total_acc(sess, model, next_batch, batch_loop, dataset.validation_set.data_size)

        self.log(
            'global_step : %04d ' % global_step,
            'loss: {:.4} '.format(loss),
            'train acc: {:.4} '.format(train_acc),
            'valid acc: {:.4} '.format(valid_acc),
        )
