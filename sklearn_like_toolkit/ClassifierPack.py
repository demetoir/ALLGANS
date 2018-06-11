import os
from pprint import pformat
from env_settting import SKLEARN_PARAMS_SAVE_PATH
from sklearn_like_toolkit.ParamOptimizer import ParamOptimizer
from sklearn_like_toolkit.Base import BaseClass
from sklearn_like_toolkit.warpper.lightGBM_wrapper import LightGBM
from sklearn_like_toolkit.warpper.sklearn_wrapper import skMLP, skSGD, skGaussian_NB, skBernoulli_NB, skMultinomial_NB, \
    skDecisionTree, skRandomForest, skExtraTrees, skAdaBoost, skGradientBoosting, skQDA, skKNeighbors, skLinear_SVC, \
    skRBF_SVM, skGaussianProcess, skVoting, skBagging
from sklearn_like_toolkit.warpper.xgboost_wrapper import XGBoost
from util.Logger import Logger
from util.misc_util import time_stamp, dump_pickle, load_pickle


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
        "skBagging": skBagging,
        "XGBoost": XGBoost,
        "LightGBM": LightGBM,
    }

    def __init__(self, pack_keys=None):
        self.log = Logger(self.__class__.__name__)
        if pack_keys is None:
            pack_keys = self.class_pack.keys()

        self.pack = {}
        for key in pack_keys:
            self.pack[key] = self.class_pack[key]()

        self.optimize_result = {}

        self.params_save_path = SKLEARN_PARAMS_SAVE_PATH

    def param_search(self, Xs, Ys):
        for key in self.pack:
            cls = self.class_pack[key]
            obj = cls()

            optimizer = ParamOptimizer(obj, obj.tuning_grid)
            self.pack[key] = optimizer.optimize(Xs, Ys)
            self.optimize_result[key] = optimizer.result

            optimizer.result_to_csv()

            self.log.info("top 5 result")
            for result in optimizer.top_k_result():
                self.log.info(pformat(result))

    def predict(self, Xs):
        result = {}
        for key in self.pack:
            result[key] = self.pack[key].predict(Xs)
        return result

    def fit(self, Xs, Ys):
        for key in self.pack:
            self.pack[key].fit(Xs, Ys)

    def score(self, Xs, Ys):
        result = {}
        for key in self.pack:
            try:
                result[key] = self.pack[key].score(Xs, Ys)
            except BaseException as e:
                self.log.error(f'while score, {key} raise {e}')

        return result

    def predict_proba(self, Xs):
        result = {}
        for key in self.pack:
            try:
                result[key] = self.pack[key].predict_proba(Xs)
            except BaseException as e:
                self.log.error(f'while predict_proba, {key} raise {e}')
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
