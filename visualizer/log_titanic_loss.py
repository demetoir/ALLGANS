from dict_keys.dataset_batch_keys import *
from visualizer.AbstractVisualizer import AbstractVisualizer


class log_titanic_loss(AbstractVisualizer):
    """visualize log of classifier's loss"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        batch_xs, batch_labels = dataset.train_set.next_batch(
            model.batch_size,
            batch_keys=[BK_X, BK_LABEL]
        )
        loss, global_step = sess.run([model.loss_mean, model.global_step],
                                     feed_dict={model.X: batch_xs, model.label: batch_labels})

        train_acc = 0.0
        for i in range(dataset.train_set.data_size // model.batch_size + 1):
            batch_xs, batch_labels = dataset.train_set.next_batch(
                model.batch_size,
                batch_keys=[BK_X, BK_LABEL]
            )
            acc = sess.run(model.batch_acc,
                           feed_dict={model.X: batch_xs, model.label: batch_labels})
            train_acc += acc
        train_acc /= (dataset.train_set.data_size // model.batch_size + 1)

        valid_acc = 0.0
        for i in range(dataset.validation_set.data_size // model.batch_size + 1):
            batch_xs, batch_labels = dataset.validation_set.next_batch(
                model.batch_size,
                batch_keys=[BK_X, BK_LABEL]
            )
            acc = sess.run(model.batch_acc,
                           feed_dict={model.X: batch_xs, model.label: batch_labels})
            valid_acc += acc
        valid_acc /= (dataset.validation_set.data_size // model.batch_size + 1)

        self.log(
            'global_step : %04d ' % global_step,
            'loss: {:.4} '.format(loss),
            'train acc: {:.4} '.format(train_acc),
            'valid acc: {:.4} '.format(valid_acc),
        )
