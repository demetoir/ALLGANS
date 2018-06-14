import progressbar
import sklearn
import pandas as pd
from data_handler.BaseDataset import BaseDataset
from util.Logger import StdoutOnlyLogger
from util.misc_util import time_stamp, setup_directory, path_join, log_error_trace
from util.numpy_utils import reformat_np_arr


class Dataset(BaseDataset):

    def load(self, path, limit=None):
        pass

    def save(self):
        pass

    def preprocess(self):
        pass


class ParamOptimizer:

    def __init__(self, estimator, param_grid=None):
        self.estimator = estimator
        if param_grid is None:
            self.param_grid = estimator.param_grid
        else:
            self.param_grid = param_grid

        self.optimizer = None
        self.result = None

        self.log = StdoutOnlyLogger(self.__class__.__name__)
        self.best_param = None
        self.best_estimator = None

    @property
    def grid_lens(self):
        grid_len = {}
        for k in sorted(self.param_grid.keys()):
            grid_len[k] = len(self.param_grid[k])

        return grid_len

    @property
    def param_grid_size(self):
        grid_len = self.grid_lens
        size = 1
        for k in grid_len:
            size *= grid_len[k]
        return size

    def get_param_by_index(self, param_grid, index):
        grid_len = self.grid_lens
        base_ = self.param_grid_size
        key_index = {}

        for key in sorted(grid_len.keys()):
            base_ //= grid_len[key]
            p = index // base_
            key_index[key] = p
            index = index % base_

        param = {}
        for k in key_index:
            param[k] = param_grid[k][key_index[k]]
        return param

    def gen_param(self):
        grid_len = self.grid_lens
        grid_size = self.param_grid_size
        sorted_keys = sorted(self.grid_lens.keys())

        for index in range(grid_size):
            _grid_size = grid_size
            key_index = {}
            for k in sorted_keys:
                _grid_size //= grid_len[k]
                p = index // _grid_size
                key_index[k] = p
                index = index % _grid_size

            param = {}
            for k in sorted_keys:
                param[k] = self.param_grid[k][key_index[k]]

            yield param

    def optimize(self, Xs, Ys, Ys_type=None, split_rate=0.7):
        Ys = reformat_np_arr(Ys, self.estimator.model_Ys_type, from_np_arr_type=Ys_type)
        dataset = Dataset()
        dataset.add_data('Xs', Xs)
        dataset.add_data('Ys', Ys)

        train_set, test_set = dataset.split((split_rate, 1 - split_rate), shuffle=False)
        train_Xs, train_Ys = train_set.full_batch()
        test_Xs, test_Ys = test_set.full_batch()

        train_Ys = reformat_np_arr(train_Ys, self.estimator.model_Ys_type, from_np_arr_type=Ys_type)
        test_Ys = reformat_np_arr(test_Ys, self.estimator.model_Ys_type, from_np_arr_type=Ys_type)

        param_grid_size = self.param_grid_size
        self.log.info(("optimize [%s], total %s's candidate estimator" % (self.estimator, param_grid_size)))
        self.result = []

        class_ = self.estimator.__class__
        gen_param = self.gen_param()
        for _ in progressbar.progressbar(range(param_grid_size), redirect_stdout=False):
            try:
                param = next(gen_param)

                estimator = class_(**param)
                estimator.fit(train_Xs, train_Ys)
                train_score = estimator.score(train_Xs, train_Ys)
                test_score = estimator.score(test_Xs, test_Ys)
                predict = estimator.predict(test_Xs)
                auc_score = sklearn.metrics.roc_auc_score(test_Ys, predict)

                result = {
                    "train_score": train_score,
                    "test_score": test_score,
                    "param": param,
                    "auc_score": auc_score,
                }
                self.result += [result]
            except KeyboardInterrupt as e:
                log_error_trace(self.log.error, e)
                raise KeyboardInterrupt
            except BaseException as e:
                self.log.warn(f'while optimize param {param} raise {e}')

        self.result = sorted(
            self.result,
            key=lambda a: (-a["auc_score"], -a["test_score"], -a["train_score"],)
        )
        self.best_param = self.result[0]["param"]
        estimator = class_(**self.best_param)
        estimator.fit(train_Xs, train_Ys, )
        return estimator

    def result_to_csv(self, path=None):
        if self.result is None:
            raise AttributeError("param optimizer does not have result")

        if path is None:
            save_path = path_join('.', 'param_search_result', time_stamp())
            setup_directory(save_path)
            file_name = str(self.estimator.__class__.__name__) + '.csv'
            path = path_join(save_path, file_name)

        pd.DataFrame(self.result).to_csv(path)
        self.log.info("param search result csv saved at %s" % path)

    def top_k_result(self, k=5):
        return self.result[:k]
