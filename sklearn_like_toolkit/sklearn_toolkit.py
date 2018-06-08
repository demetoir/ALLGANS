import os
from pprint import pprint, pformat
import sklearn
import progressbar

from env_settting import SKLEARN_PARAMS_SAVE_PATH
from sklearn_like_toolkit.base import BaseSklearn, BaseClass
from sklearn_like_toolkit.lightGBM_wrapper import LightGBM
from sklearn_like_toolkit.sklearn_wrapper import skMLP, skSGD, skGaussian_NB, skBernoulli_NB, skMultinomial_NB, \
    skDecisionTree, skRandomForest, skExtraTrees, skAdaBoost, skGradientBoosting, skQDA, skKNeighbors, skLinear_SVC, \
    skRBF_SVM, skGaussianProcess
from sklearn_like_toolkit.xgboost_wrapper import XGBoost
from util.Logger import StdoutOnlyLogger
from util.misc_util import time_stamp, dump_pickle, load_pickle, setup_directory, path_join
from util.numpy_utils import reformat_np_arr
from data_handler.BaseDataset import BaseDataset


class Dataset(BaseDataset):

    def load(self, path, limit=None):
        pass

    def save(self):
        pass

    def preprocess(self):
        pass


class ParamOptimizer(BaseSklearn):

    def __init__(self, estimator, param_grid=None, **kwargs):
        super().__init__(estimator, param_grid, **kwargs)
        self.estimator = estimator
        if param_grid is None:
            self.param_grid = estimator.param_grid
        else:
            self.param_grid = param_grid

        self.optimizer = None
        self.result = None

        self.logger = StdoutOnlyLogger(self.__class__.__name__)
        self.log = self.logger.get_log()
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

    # def optimize(self, Xs, Ys, Ys_type=None, split_rate=0.7):
    #     Ys = reformat_np_arr(Ys, self.estimator.model_Ys_type, from_np_arr_type=Ys_type)
    #     dataset = Dataset()
    #     dataset.add_data('Xs', Xs)
    #     dataset.add_data('Ys', Ys)
    #
    #     train_set, test_set = dataset.split((0.7, 0.3), shuffle=False)
    #     train_Xs, train_Ys = train_set.full_batch()
    #     test_Xs, test_Ys = test_set.full_batch()

    def optimize(self, train_Xs, train_Ys, test_Xs, test_Ys, Ys_type=None):
        train_Ys = reformat_np_arr(train_Ys, self.estimator.model_Ys_type, from_np_arr_type=Ys_type)
        test_Ys = reformat_np_arr(test_Ys, self.estimator.model_Ys_type, from_np_arr_type=Ys_type)

        param_grid_size = self.param_grid_size
        self.log(("optimize [%s], total %s's candidate estimator" % (self.estimator, param_grid_size)))
        self.result = []

        class_ = self.estimator.__class__
        gen_param = self.gen_param()
        for _ in progressbar.progressbar(range(param_grid_size), redirect_stdout=False):
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

        self.result = sorted(
            self.result,
            key=lambda a: (-a["auc_score"], -a["test_score"], -a["train_score"],)
        )
        self.best_param = self.result[0]["param"]
        estimator = class_(**self.best_param)
        estimator.fit(train_Xs, train_Ys)
        return estimator

    def result_to_csv(self, path):
        if self.result is None:
            raise AttributeError("param optimizer does not have result")

        import pandas as pd

        result = self.result
        for i in range(len(self.result)):
            result[i].update(self.result[i]['param'])
            result[i].pop('param')

        df = pd.DataFrame(result)
        df.to_csv(path)

    def top_k_result(self, k=5):
        return self.result[:k]


class ClassifierPack(BaseClass):
    class_pack = {
        "skMLP": skMLP,
        "skSGD": skSGD,
        "skGaussian_NB": skGaussian_NB,
        "skBernoulli_NB": skBernoulli_NB,
        "skMultinomial_NB": skMultinomial_NB,
        "skDecisionTree": skDecisionTree,
        "skRandomForest": skRandomForest,
        "skExtraTrees": skExtraTrees,
        "skAdaBoost": skAdaBoost,
        "skGradientBoosting": skGradientBoosting,
        "skQDA": skQDA,
        "skKNeighbors": skKNeighbors,
        "skLinear_SVC": skLinear_SVC,
        "skRBF_SVM": skRBF_SVM,
        "skGaussianProcess": skGaussianProcess,
        "XGBoost": XGBoost,
        "LightGBM": LightGBM,
    }

    def __init__(self, pack_keys=None):
        if pack_keys is None:
            pack_keys = self.class_pack.keys()

        self.pack = {}
        for key in pack_keys:
            cls = self.class_pack[key]
            obj = cls()
            self.pack[key] = obj

        self.logger = StdoutOnlyLogger(self.__class__.__name__)
        self.log = self.logger

        self.params_save_path = SKLEARN_PARAMS_SAVE_PATH

    def param_search(self, train_Xs, train_Ys, test_Xs, test_Ys):
        save_path = path_join('.', 'param_search_result', time_stamp())
        setup_directory(save_path)

        for key in self.pack:
            cls = self.class_pack[key]
            obj = cls()

            optimizer = ParamOptimizer(obj, obj.tuning_grid)
            self.pack[key] = optimizer.optimize(train_Xs, train_Ys, test_Xs, test_Ys)

            file_name = str(obj) + '.csv'
            csv_path = path_join(save_path, file_name)
            self.log.info("param search result csv saved at %s" % csv_path)
            optimizer.result_to_csv(csv_path)

            self.log.info("top 5 result")
            for result in optimizer.top_k_result():
                self.log.info(pformat(result))

    def predict(self, Xs):
        result = {}
        for key in self.pack:
            clf = self.pack[key]
            result[key] = clf.predict(Xs)
        return result

    def fit(self, Xs, Ys, Ys_type=None):
        for key in self.pack:
            clf = self.pack[key]
            clf.fit(Xs, Ys, Ys_type=Ys_type)

    def score(self, Xs, Ys, Ys_type=None):
        result = {}
        for key in self.pack:
            clf = self.pack[key]
            result[key] = clf.score(Xs, Ys, Ys_type=Ys_type)
        return result

    def proba(self, Xs, transpose_shape=True):
        result = {}
        for key in self.pack:
            clf = self.pack[key]
            result[key] = clf.proba(Xs, transpose_shape=transpose_shape)
        return result

    def import_params(self, params_pack):
        for key in self.pack:
            class_ = self.class_pack[key]
            self.pack[key] = class_(**params_pack[key])

    def export_params(self):
        params = {}
        for key in self.pack:
            clf = self.pack[key]
            params[key] = clf.get_params()
        return params

    def save_params(self, path=None):

        if path is None:
            path = os.path.join(self.params_save_path, time_stamp())

        params = self.export_params()

        pickle_path = path + '.pkl'
        dump_pickle(params, pickle_path)

        self.log.info('save params at {}'.format([pickle_path]))

        return pickle_path

    def load_params(self, path):
        self.log.info('load params from {}'.format(path))
        params = load_pickle(path)

        self.import_params(params)
