from dict_keys.dataset_batch_keys import *
from visualizer.AbstractVisualizer import AbstractVisualizer
import pandas as pd
import os
import numpy as np


class csv_titanic_result(AbstractVisualizer):
    def task(self, sess=None, iter_num=None, model=None, dataset=None):
        dataset.test_set.reset_cursor()
        batch_loop = dataset.test_set.data_size // model.batch_size + 1
        predict_list = []
        for i in range(batch_loop):
            batch_xs = dataset.test_set.next_batch(
                model.batch_size,
                batch_keys=[BK_X]
            )
            predict = sess.run(
                model.predict_index,
                feed_dict={
                    model.X: batch_xs,
                    model.dropout_rate: 1
                }
            )
            predict_list += [predict]
        predict = np.concatenate(predict_list, axis=0)[:dataset.test_set.data_size]
        predict = predict.astype(np.int32)

        df = pd.DataFrame()
        df["PassengerId"] = range(892, 1309 + 1)
        df["Survived"] = predict

        path = os.path.join(self.visualizer_path, "result.csv")
        df.to_csv(path_or_buf=path, index=False, index_label=False)

        self.log("total sample %s, build result.csv file\n%s" % (dataset.test_set.data_size, path))
