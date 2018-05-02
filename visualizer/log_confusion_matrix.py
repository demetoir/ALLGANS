from dict_keys.dataset_batch_keys import *
from visualizer.AbstractVisualizer import AbstractVisualizer
import numpy as np


class log_confusion_matrix(AbstractVisualizer):
    """visualize confusion matrix by print log"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        matrix = np.zeros([dataset.LABEL_SIZE, dataset.LABEL_SIZE], dtype=np.int32)

        dataset.validation_set.reset_cursor()
        batch_loop = dataset.validation_set.data_size // model.batch_size + 1
        hs_list = []

        for _ in range(batch_loop):
            batch_x, batch_labels, ids = dataset.validation_set.next_batch(model.batch_size,
                                                                           batch_keys=[BK_X, BK_LABEL, "PassengerId"])
            predict_labels, true_labels, hs = sess.run(
                [model.predict_index, model.label_index, model.h],
                feed_dict={
                    model.X: batch_x,
                    model.label: batch_labels,
                    model.dropout_rate: 1
                }
            )

            for true_label, predict_label, h, id_ in zip(true_labels, predict_labels, hs, ids):
                matrix[int(true_label - 1)][int(predict_label - 1)] += 1
                if true_label != predict_label:
                    hs_list += [h]
                    # self.log("id: %s, h: %s, true label: %s" % (id_, h, true_label))

        # TODO better look
        msg = "confusion matrix total %d samples\n%s\n\n%s" \
              % (model.batch_size * batch_loop,
                 str(np.round(matrix, decimals=4)),
                 str(np.round(matrix / (model.batch_size * batch_loop), decimals=4))
                 )

        self.log(msg)
