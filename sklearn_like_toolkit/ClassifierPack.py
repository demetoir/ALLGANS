import os
from pprint import pformat
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from env_settting import SKLEARN_PARAMS_SAVE_PATH
from sklearn_like_toolkit.Base import BaseClass
from sklearn_like_toolkit.FoldingHardVote import FoldingHardVote
from sklearn_like_toolkit.ParamOptimizer import ParamOptimizer
from sklearn_like_toolkit.warpper.catboost_wrapper import CatBoostClf
from sklearn_like_toolkit.warpper.lightGBM_wrapper import LightGBMClf
from sklearn_like_toolkit.warpper.mlxtend_wrapper import mlxAdalineClf, mlxLogisticRegressionClf, \
    mlxMLPClf, mlxPerceptronClf, mlxSoftmaxRegressionClf, mlxStackingCVClf, mlxStackingClf
from sklearn_like_toolkit.warpper.sklearn_wrapper import skMLP, skSGD, skGaussian_NB, skBernoulli_NB, skMultinomial_NB, \
    skDecisionTree, skRandomForest, skExtraTrees, skAdaBoost, skGradientBoosting, skQDA, skKNeighbors, skLinear_SVC, \
    skRBF_SVM, skGaussianProcess, skBagging
from sklearn_like_toolkit.warpper.xgboost_wrapper import XGBoostClf
from util.Logger import Logger
from util.misc_util import time_stamp, dump_pickle, load_pickle
from util.numpy_utils import reformat_np_arr, NP_ARRAY_TYPE_INDEX, NP_ARRAY_TYPE_ONEHOT

CLF_METRICS = {
    'accuracy': accuracy_score,
    'confusion_matrix': confusion_matrix,
    'roc_auc_score': roc_auc_score,
    'recall_score': recall_score,
    'precision_score': precision_score,
}


class _Reformat_Ys:

    def _reformat_to_index(self, Xs):
        return reformat_np_arr(Xs, NP_ARRAY_TYPE_INDEX)

    def _reformat_to_onehot(self, Xs):
        return reformat_np_arr(Xs, NP_ARRAY_TYPE_ONEHOT)


class ClassifierPack(BaseClass, _Reformat_Ys):
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
        "XGBoost": XGBoostClf,
        "LightGBM": LightGBMClf,
        "CatBoost": CatBoostClf,
        'mlxAdaline': mlxAdalineClf,
        'mlxLogisticRegression': mlxLogisticRegressionClf,
        'mlxMLP': mlxMLPClf,
        'mlxPerceptronClf': mlxPerceptronClf,
        'mlxSoftmaxRegressionClf': mlxSoftmaxRegressionClf,
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

        self._metrics = CLF_METRICS

    def param_search(self, Xs, Ys):
        Ys = self._reformat_to_index(Ys)
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

    def _collect_predict(self, Xs):
        result = {}
        for key in self.pack:
            try:
                result[key] = self.pack[key].predict(Xs)
            except BaseException as e:
                self.log.warn(f'while fitting, {key} raise {e}')
        return result

    def predict(self, Xs):
        return self._collect_predict(Xs)

    def fit(self, Xs, Ys):
        Ys = self._reformat_to_index(Ys)
        for key in self.pack:
            try:
                self.pack[key].fit(Xs, Ys)
            except BaseException as e:
                self.log.warn(f'while fitting, {key} raise {e}')

    def _apply_metric(self, Y_true, Y_predict, metric):
        return self._metrics[metric](Y_true, Y_predict)

    def score(self, Xs, Ys, metric='accuracy'):
        Ys = self._reformat_to_index(Ys)
        scores = {}
        for clf_k, predict in self._collect_predict(Xs).items():
            scores[clf_k] = self._apply_metric(Ys, predict, metric)
        return scores

    def score_pack(self, Xs, Ys):
        Ys = self._reformat_to_index(Ys)
        ret = {}
        for clf_k, predict in self._collect_predict(Xs).items():
            ret[clf_k] = {metric: self._apply_metric(Ys, predict, metric) for metric in self._metrics}
        return ret

    def predict_proba(self, Xs):
        result = {}
        for key in self.pack:
            try:
                result[key] = self.pack[key].predict_proba(Xs)
            except BaseException as e:
                self.log.warn(f'while predict_proba, {key} raise {e}')
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

    def make_FoldingHardVote(self):
        clfs = [v for k, v in self.pack.items()]
        return FoldingHardVote(clfs)

    def make_stackingClf(self, meta_clf):
        clfs = [clf for k, clf in self.pack.items() if hasattr(clf, 'get_params')]
        return mlxStackingClf(clfs, meta_clf)

    def make_stackingCVClf(self, meta_clf):
        clfs = [clf for k, clf in self.pack.items() if hasattr(clf, 'get_params')]
        return mlxStackingCVClf(clfs, meta_clf)

    def clone_top_k_tuned(self, k=5):
        new_pack = {}
        for key in self.pack:
            new_pack[key] = self.pack[key]
            results = self.optimize_result[key][1:k]

            for i, result in enumerate(results):
                param = result["param"]
                cls = self.pack[key].__class__
                new_key = str(cls.__name__) + '_' + str(i + 1)
                clf = cls(**param)
                new_pack[new_key] = clf

        self.pack = new_pack
        return self.pack

    def drop_clf(self, key):
        self.pack.pop(key)

    def add_clf(self, key, clf):
        if key in self.pack:
            raise KeyError(f"key '{key}' is not unique")

        self.pack[key] = clf

    def clone_clf(self, key, n=1, param=None):
        if key not in self.pack:
            raise KeyError(f"key '{key}' not exist")

        # check_origin()
        #
        # for i in range(n):
        #     clone()
