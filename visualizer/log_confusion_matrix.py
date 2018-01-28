from visualizer.AbstractPrintLog import AbstractPrintLog
import numpy as np

from dict_keys.dataset_batch_keys import *


class print_confusion_matrix(AbstractPrintLog):
    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        # test data
        matrix = np.zeros([dataset.LABEL_SIZE, dataset.LABEL_SIZE], dtype=np.int32)

        for _ in range(10):
            batch_x, batch_labels = dataset.next_batch(model.batch_size,
                                                       batch_keys=[BATCH_KEY_TEST_X, BATCH_KEY_TEST_LABEL])
            predict_labels, true_labels = sess.run([model.predict_index, model.label_index],
                                                   feed_dict={model.X: batch_x, model.label: batch_labels})

            for true_label, predict_label in zip(true_labels, predict_labels):
                matrix[int(true_label - 1)][int(predict_label - 1)] += 1

        # TODO better look
        self.log("confusion matrix total %d samples\n" % (model.batch_size * 10) + str(matrix))
