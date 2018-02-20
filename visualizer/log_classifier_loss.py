from dict_keys.dataset_batch_keys import *
from visualizer.AbstractVisualizer import AbstractVisualizer


class log_classifier_loss(AbstractVisualizer):
    """visualize log of classifier's loss"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        batch_xs, batch_labels = dataset.next_batch(model.batch_size,
                                                    batch_keys=[BATCH_KEY_TRAIN_X, BATCH_KEY_TRAIN_LABEL])
        loss, train_acc, global_step = sess.run([model.loss_mean, model.batch_acc, model.global_step],
                                                feed_dict={model.X: batch_xs, model.label: batch_labels})

        batch_xs, batch_labels = dataset.next_batch(model.batch_size,
                                                    batch_keys=[BATCH_KEY_TEST_X, BATCH_KEY_TEST_LABEL])
        test_acc = sess.run(model.batch_acc, feed_dict={model.X: batch_xs, model.label: batch_labels})

        self.log(
            'global_step : %04d ' % global_step,
            'loss: {:.4} '.format(loss),
            'train acc: {:.4} '.format(train_acc),
            'test acc: {:.4} '.format(test_acc),
        )
