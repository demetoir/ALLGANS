from dict_keys.dataset_batch_keys import *
from visualizer.AbstractVisualizer import AbstractVisualizer
import numpy as np


class log_confusion_matrix(AbstractVisualizer):
    """visualize confusion matrix by print log"""

    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        matrix = np.zeros([dataset.LABEL_SIZE, dataset.LABEL_SIZE], dtype=np.int32)

        for _ in range(1):
            batch_x, batch_labels = dataset.validation_set.next_batch(model.batch_size, batch_keys=[BK_X, BK_LABEL])
            predict_labels, true_labels = sess.run([model.predict_index, model.label_index],
                                                   feed_dict={model.X: batch_x, model.label: batch_labels, model.dropout_rate:1})

            for true_label, predict_label in zip(true_labels, predict_labels):
                matrix[int(true_label - 1)][int(predict_label - 1)] += 1
                # if int(true_label - 1) != int(predict_label - 1):


        # TODO better look
        self.log("confusion matrix total %d samples\n" % (model.batch_size * 1) + str(matrix))
