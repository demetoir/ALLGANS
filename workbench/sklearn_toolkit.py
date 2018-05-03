from pprint import pprint

import numpy as np
import sklearn
from sklearn.gaussian_process.kernels import RBF
from util.numpy_utils import np_onehot_to_index, np_index_to_onehot

YS_TYPE_INDEX = "index"
YS_TYPE_ONEHOT = "onehot"
INDEX_TO_ONEHOT = (YS_TYPE_INDEX, YS_TYPE_ONEHOT)
ONEHOT_TO_INDEX = (YS_TYPE_ONEHOT, YS_TYPE_INDEX)
NO_CONVERT = (
    (YS_TYPE_ONEHOT, YS_TYPE_ONEHOT),
    (YS_TYPE_INDEX, YS_TYPE_INDEX)
)


def reformat_Ys(Ys, model_Ys_type, Ys_type=None):
    def get_Ys_type(Ys):
        shape = Ys.shape
        if len(shape) == 1:
            _type = "index"
        elif len(shape) == 2:
            _type = "onehot"
        else:
            _type = "invalid"

        return _type

    if Ys_type is None:
        Ys_type = get_Ys_type(Ys)

    convert_type = (Ys_type, model_Ys_type)

    if convert_type == ONEHOT_TO_INDEX:
        Ys = np_onehot_to_index(Ys)
    elif convert_type == INDEX_TO_ONEHOT:
        Ys = np_index_to_onehot(Ys)
    elif convert_type in NO_CONVERT:
        pass
    else:
        raise TypeError("Ys convert type Error Ys_type=%s, model_Ys_type=%s" % (Ys_type, model_Ys_type))

    return Ys


def print_dict(d):
    for k in d:
        print("'%s' : %s," % (k, d[k]))


class BaseClass:
    def __str__(self):
        return self.__class__.__name__


class BaseSklearn(BaseClass):
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, Xs, Ys, Ys_type=None):
        pass

    def predict(self, Xs):
        pass

    def score(self, Xs, Ys, Ys_type):
        pass

    def proba(self, Xs, transpose_shape=True):
        pass

    def get_params(self, deep=True):
        pass

    def set_params(self, **params):
        pass


class ParamGridSearch(BaseSklearn):

    def __init__(self, estimator, param_grid=None, **kwargs):
        super().__init__(estimator, param_grid, **kwargs)
        self.estimator = estimator
        if param_grid is None:
            self.param_grid = estimator.param_grid
        else:
            self.param_grid = param_grid

        from sklearn.model_selection import GridSearchCV as _GridSearchCV
        self.param_search = _GridSearchCV(self.estimator, self.param_grid, **kwargs)
        del _GridSearchCV

        # print_dict(self.param_search.__dict__)

    @property
    def cv_results_(self):
        return self.param_search.cv_results_

    @property
    def best_estimator(self):
        return self.param_search.best_estimator_

    @property
    def best_params_(self):
        return self.param_search.best_params_

    @property
    def best_score_(self):
        return self.param_search.best_score_

    def fit(self, Xs, Ys, Ys_type=None):
        Ys = reformat_Ys(Ys, self.estimator.model_Ys_type, Ys_type=Ys_type)
        self.param_search.fit(Xs, Ys)

    def predict(self, Xs):
        super().predict(Xs)

    def score(self, Xs, Ys, Ys_type):
        super().score(Xs, Ys, Ys_type)

    def proba(self, Xs, transpose_shape=True):
        super().proba(Xs, transpose_shape)

    def get_params(self, deep=True):
        return self.param_search.get_params(deep=deep)

    def set_params(self, **params):
        return self.param_search.set_params(**params)


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

    def optimize(self, train_Xs, test_Xs, train_Ys, test_Ys, Ys_type=None):
        train_Ys = reformat_Ys(train_Ys, self.estimator.model_Ys_type, Ys_type=Ys_type)
        test_Ys = reformat_Ys(test_Ys, self.estimator.model_Ys_type, Ys_type=Ys_type)

        param_grid_size = self.param_grid_size
        print("total %s's candidate estimator" % param_grid_size)
        self.result = []

        class_ = self.estimator.__class__
        estimator = self.estimator
        for idx, param in enumerate(self.gen_param()):
            print("%s/%s fitting model" % (idx + 1, param_grid_size))

            # estimator = class_(**param)
            estimator.set_params(**param)
            # print(estimator.get_params())
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

            print(train_score)
            print(test_score)
            print(auc_score)

        comp = lambda a: (-a["test_score"], -a["train_score"], -a["auc_score"])

        self.result = sorted(self.result, key=comp)
        self.best_param = self.result[0]["param"]

        print("best 5")
        for d in self.result[:5]:
            pprint(d)

    def result_to_csv(self, path):
        if self.result is None:
            raise AttributeError("param optimizer does not have result")

        import pandas as pd

        df = pd.DataFrame(self.result)
        df.to_csv(path)


class BaseSklearnClassifier(BaseSklearn):
    model_Ys_type = None
    tuning_params = None
    tuning_grid = None

    def __str__(self):
        return super().__str__() + "\ntuning params\n" + \
               self.tuning_params.__str__()

    def __init__(self, **params):
        super().__init__(**params)
        self.model = None

    def fit(self, Xs, Ys, Ys_type=None):
        self.model.fit(Xs, reformat_Ys(Ys, self.model_Ys_type, Ys_type=Ys_type))

    def predict(self, Xs):
        return self.model.predict(Xs)

    def score(self, Xs, Ys, Ys_type=None):
        return self.model.score(Xs, reformat_Ys(Ys, self.model_Ys_type, Ys_type=Ys_type))

    def proba(self, Xs, transpose_shape=True):
        """
        if multi label than output shape == (class, sample, prob)
        need to transpose shape to (sample, class, prob)

        :param Xs:
        :param transpose_shape:
        :return:
        """
        probs = np.array(self.model.predict_proba(Xs))

        if transpose_shape is True:
            probs = np.transpose(probs, axes=(1, 0, 2))

        return probs

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        return self.model.set_params(**params)


class MLP(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_ONEHOT
    tuning_grid = {
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'alpha': [0.01, 0.1, 1, 10],
        'hidden_layer_sizes': [(32,), (64,), (128,)],
    }
    tuning_params = {
        'hidden_layer_sizes': (100,),
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'max_iter': 200,
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'learning_rate_init': 0.001,
        'tol': 0.0001,
        # L2 penalty
        'alpha': 1,
    }
    solver_param = {
        'solver': ['lbfgs', 'sgd', 'adam'],

        # adam solver option
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-08,

        # sgd solver option
        # between 0 and 1.
        'momentum': 0.9,
        'nesterovs_momentum': True,
        'power_t': 0.5,
    }
    etc_param = {
        'random_state': None,
        'verbose': False,
        'warm_start': False,
        'early_stopping': False,

        # batch option
        'batch_size': 'auto',
        'shuffle': True,
        'validation_fraction': 0.1,

    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.neural_network import MLPClassifier as _MLP
        self.model = _MLP(alpha=1)
        del _MLP


class Gaussian_NB(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX
    tuning_grid = {}
    tuning_params = {
        'priors': None
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.naive_bayes import GaussianNB as _GaussianNB
        self.model = _GaussianNB(**params)
        del _GaussianNB


class Bernoulli_NB(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX
    tuning_grid = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        'binarize': [i / 10.0 for i in range(0, 10)],
    }
    tuning_params = {
        'alpha': 1.0,
        'binarize': 0.0,
        'class_prior': None,
        'fit_prior': True
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.naive_bayes import BernoulliNB as _BernoulliNB
        self.model = _BernoulliNB(**params)
        del _BernoulliNB


class Multinomial_NB(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX
    tuning_grid = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    }
    tuning_params = {
        'alpha': 1.0,
        'class_prior': None,
        'fit_prior': True
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.naive_bayes import MultinomialNB as _MultinomialNB
        self.model = _MultinomialNB(**params)
        del _MultinomialNB


class QDA(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX
    tuning_grid = {

    }
    tuning_params = {

    }
    param = {
        # TODO
        # ? ..
        'priors': None,
        'reg_param': 0.0,
        'store_covariance': False,
        'store_covariances': None,
        'tol': 0.0001
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as _QDA
        self.model = _QDA(**params)
        del _QDA


class DecisionTree(BaseSklearnClassifier):
    """
    sklearn base DecisionTreeClassifier
    """
    model_Ys_type = YS_TYPE_ONEHOT
    tuning_grid = {
        'max_depth': [i for i in range(1, 10)],
        'min_samples_leaf': [i for i in range(1, 5)],
        'min_samples_split': [i for i in range(2, 5)],
    }
    tuning_params = {
        'max_depth': None,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
    }
    default_only_params = {
        'criterion': 'gini',
        'splitter': 'best',
        'max_leaf_nodes': None,
        'max_features': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
    }
    etc_param = {
        # class weight options
        'class_weight': None,
        'min_weight_fraction_leaf': 0.0,

        'presort': False,
        'random_state': None,
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.tree import DecisionTreeClassifier as _DecisionTreeClassifier
        self.model = _DecisionTreeClassifier(**params)
        del _DecisionTreeClassifier


class RandomForest(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_ONEHOT
    tuning_grid = {
        'n_estimators': [2, 4, 8, 16, 32, 64],
        'max_depth': [i for i in range(1, 10)],
        'min_samples_leaf': [i for i in range(1, 5)],
        'min_samples_split': [i for i in range(2, 5)],
    }
    tuning_params = {
        'n_estimators': 10,
        'max_depth': None,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
    }
    etc_param = {
        # class weight option
        'class_weight': None,
        'min_weight_fraction_leaf': 0.0,

        # etc
        'n_jobs': 1,
        'oob_score': False,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
    }
    default_only_params = {
        'max_features': 'auto',
        'criterion': 'gini',
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
        'max_leaf_nodes': None,
        'bootstrap': True,
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier
        self.model = _RandomForestClassifier(**params)
        del _RandomForestClassifier


class ExtraTrees(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_ONEHOT
    tuning_grid = {
        'n_estimators': [2, 4, 8, 16, 32, 64],
        'max_depth': [i for i in range(1, 10)],
        'min_samples_leaf': [i for i in range(1, 5)],
        'min_samples_split': [i for i in range(2, 5)],
    }
    tuning_params = {
        # tuning params
        'n_estimators': 10,
        'max_depth': None,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
    }
    only_default_params = {
        'bootstrap': False,
        'criterion': 'gini',
        'max_features': 'auto',
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
    }
    etc_param = {
        # class weight options
        'class_weight': None,
        'min_weight_fraction_leaf': 0.0,

        # etc
        'n_jobs': 1,
        'oob_score': False,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.ensemble import ExtraTreesClassifier as _ExtraTreesClassifier

        params.update(self.etc_param)
        # print(params)
        self.model = _ExtraTreesClassifier(**params)
        del _ExtraTreesClassifier


class AdaBoost(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX
    tuning_grid = {
        'learning_rate': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10],
        'n_estimators': [2, 4, 8, 16, 32, 64, 128, 256],
    }
    tuning_params = {
        'base_estimator': None,
        # tuning param
        'learning_rate': 1.0,
        'n_estimators': 50,

    }
    etc_param = {
        # etc
        'random_state': None,
        'algorithm': ['SAMME.R', 'SAMME'],
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.ensemble import AdaBoostClassifier as _AdaBoostClassifier
        self.model = _AdaBoostClassifier(**params)
        del _AdaBoostClassifier


class GradientBoosting(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX
    tuning_grid = {
        'learning_rate': [0.001, 0.01, 0.1, 1],
        'max_depth': [i for i in range(1, 10)],
        'n_estimators': [2, 4, 8, 16, 32, 64, 128, 256],
        'min_samples_leaf': [i for i in range(1, 5)],
        'min_samples_split': [i for i in range(2, 5)],
    }
    tuning_params = {
        # tuning param
        'learning_rate': 0.1,
        'max_depth': 3,
        'n_estimators': 100,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
    }
    only_default_params = {
        'criterion': 'friedman_mse',
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
        'max_features': None,
        'max_leaf_nodes': None,
    }
    etc_param = {
        # todo wtf?
        'init': None,
        'loss': ['deviance', 'exponential'],
        'subsample': 1.0,

        # etc
        'min_weight_fraction_leaf': 0.0,
        'presort': 'auto',
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.ensemble import GradientBoostingClassifier as _GradientBoostingClassifier
        self.model = _GradientBoostingClassifier(**params)
        del _GradientBoostingClassifier


class KNeighbors(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX
    tuning_grid = {
        'n_neighbors': [i for i in range(1, 32)],
    }
    tuning_params = {
        'n_neighbors': 5,
    }
    only_default_params = {
        'weights': 'uniform',
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': 2,
        'leaf_size': 30,
    }
    etc_param = {
        'n_jobs': 1,
        'metric': 'minkowski',
        'metric_params': None,
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier
        self.model = sklearn.neighbors.KNeighborsClassifier(**params)
        del _KNeighborsClassifier


class Linear_SVC(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX
    tuning_grid = {
        'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10],
        'max_iter': [2 ** i for i in range(6, 13)],
    }
    tuning_params = {
        'C': 1.0,
        'max_iter': 1000,
    }
    only_default_params = {
        'fit_intercept': True,
        'intercept_scaling': 1,

        # todo ???
        'multi_class': ['ovr', 'crammer_singer'],
        'loss': ['squared_hinge', 'hinge'],
        'penalty': ['l2', 'l1'],
        'class_weight': None,
        'dual': True,

    }
    etc_param = {
        'random_state': None,
        'tol': 0.0001,
        'verbose': 1e-4,
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.svm import LinearSVC as _LinearSVC
        self.model = _LinearSVC(**params)
        del _LinearSVC

    def proba(self, Xs, transpose_shape=True):
        print("""linear_SVM does not have attr "prob" """)
        return None


class RBF_SVM(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX
    tuning_grid = {
        'C': [1 ** i for i in range(-5, 5)],
        'gamma': [1 ** i for i in range(-5, 5)],
    }
    tuning_params = {
        'C': 1,
        'gamma': 2,
    }
    # todo
    param = {
        'cache_size': 200,
        'class_weight': None,
        'coef0': 0.0,
        'decision_function_shape': 'ovr',
        'degree': 3,
        'kernel': 'rbf',
        'max_iter': -1,
        'probability': False,
        'random_state': None,
        'shrinking': True,
        'tol': 0.001,
        'verbose': False
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.svm import SVC as _SVC
        # TODO **params
        self.model = _SVC(gamma=2, C=1)
        del _SVC

    def proba(self, Xs, transpose_shape=True):
        print("""linear_SVM does not have attr "prob" """)
        return None


class GaussianProcess(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX
    # todo
    tuning_grid = {

    }
    tuning_params = {

    }
    param = {
        'kernel': 1 ** 2 * RBF(length_scale=1),
        'kernel__k1': 1 ** 2,
        'kernel__k1__constant_value': 1.0,
        'kernel__k1__constant_value_bounds': (1e-05, 100000.0),
        'kernel__k2': RBF(length_scale=1),
        'kernel__k2__length_scale': 1.0,
        'kernel__k2__length_scale_bounds': (1e-05, 100000.0),
        'max_iter_predict': 100,

        'multi_class': 'one_vs_rest',
        'n_jobs': 1,
        'n_restarts_optimizer': 0,
        'optimizer': 'fmin_l_bfgs_b',
        'random_state': None,
        'warm_start': False,
        'copy_X_train': True,
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.gaussian_process import GaussianProcessClassifier as _GaussianProcessClassifier
        # TODO **params
        self.model = _GaussianProcessClassifier(1.0 * RBF(1.0))
        del _GaussianProcessClassifier


class SGD(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX

    # todo wtf?
    # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
    tuning_grid = {
        # todo random..
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 1.0],

    }
    tuning_params = {
        'alpha': 0.0001,
    }
    params = {
        # TODO
        'tol': None,
        'learning_rate': ['optimal', 'constant', 'invscaling'],

        'alpha': 0.0001,

        'average': False,
        'class_weight': None,
        'epsilon': 0.1,
        'eta0': 0.0,
        'fit_intercept': True,
        'l1_ratio': 0.15,
        'loss': 'hinge',
        'max_iter': None,
        'n_iter': None,

        'penalty': ['none', 'l1', 'l2', 'elasticnet'],

        'power_t': 0.5,

        # etc
        'n_jobs': 1,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,
        'shuffle': True,
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.linear_model import SGDClassifier as _SGDClassifier
        self.model = _SGDClassifier(**params)
        del _SGDClassifier

    def proba(self, Xs, transpose_shape=True):
        # todo
        return None
        # return super().proba(Xs, transpose_shape)


class XGBoost(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX
    tuning_grid = {
        'max_depth': [i for i in range(3, 10)],
        'n_estimators': [128, 256, 512, 1024, 2048],
        'min_child_weight': [1, 2, 3],
        'gamma': [i / 10.0 for i in range(0, 10, 3)],
        'subsample': [i / 10.0 for i in range(1, 10, 3)],
        'colsample_bytree': [i / 10.0 for i in range(1, 10, 3)],
        'learning_rate': [0.01, 0.1, 1],
    }
    tuning_params = {
        'max_depth': 3,
        'n_estimators': 100,
        'min_child_weight': 1,
        'gamma': 0,

        'subsample': 1,
        'colsample_bytree': 1,
        'learning_rate': 0.1,

    }
    param = {
        'silent': True,
        'objective': 'binary:logistic',
        'booster': ['gbtree', 'gblinear', 'dart'],
        'colsample_bylevel': 1,

        'reg_alpha': 0,
        'reg_lambda': 1,

        'scale_pos_weight': 1,
        'max_delta_step': 0,

        'base_score': 0.5,
        'n_jobs': 1,
        'nthread': None,
        'random_state': 0,
        'seed': None,
        'missing': None,
    }

    def __init__(self, **params):
        super().__init__(**params)

        # params.update({"tree_method": 'gpu_hist'})
        # params.update({"tree_method": 'hist'})
        # params.update({"tree_method": 'exact'})
        # params.update({"tree_method": 'gpu_exact'})

        # params.update({'nthread': 1})
        # params.update({"silent": 1})
        import xgboost as XGB
        self.model = XGB.XGBClassifier(**params)
        del XGB


class LightGBM(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX
    tuning_grid = {
        'num_leaves': [4, 8, 16, 32],
        'min_child_samples': [4, 8, 16, 32],
        'max_depth': [2, 4, 6, 8],
        'colsample_bytree': [i / 10.0 for i in range(2, 10 + 1, 2)],
        'subsample': [i / 10.0 for i in range(2, 10 + 1, 2)],
        # 'max_bin': [64, 128],
        # 'top_k': [8, 16, 32],
    }
    tuning_params = {

    }
    param = {
        'learning_rate': 0.1,
        # 'num_boost_round': 100,
        # ???
        'max_delta_step': 0,
        'min_split_gain': 0,

        # device option
        # 'device': 'gpu',
        'device': 'cpu',

        # dart only option
        'drop_rate': 0.1,
        'skip_drop': 0.5,
        'max_drop': 50,
        'uniform_drop': False,
        'xgboost_dart_mode': False,
        'drop_seed': 4,

        # goss only option
        'other_rate': 0.1,
        'min_data_per_group': 100,

        # default value
        'feature_fraction_seed': 2,
        'bagging_seed': 3,
        # 'early_stopping_round': 0,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'max_cat_threshold': 32,
        'cat_smooth': 10,
        'cat_l2': 10,
        'max_cat_to_onehot': 4,
        'verbose': -1

    }

    def __init__(self, **params):
        super().__init__(**params)

        import lightgbm as lgbm
        params.update(self.param)
        self.model = lgbm.LGBMClassifier(**params)
        del lgbm


class ClassifierPack(BaseClass):
    SGD = SGD
    Gaussian_NB = Gaussian_NB
    Bernoulli_NB = Bernoulli_NB
    Multinomial_NB = Multinomial_NB
    DecisionTree = DecisionTree
    RandomForest = RandomForest
    ExtraTrees = ExtraTrees
    AdaBoost = AdaBoost
    GradientBoosting = GradientBoosting
    MLP = MLP
    QDA = QDA
    KNeighbors = KNeighbors
    Linear_SVC = Linear_SVC
    RBF_SVM = RBF_SVM
    GaussianProcess = GaussianProcess
    XGBoost = XGBoost
    LightGBM = LightGBM
    clf_pack = [
        MLP,
        SGD,
        Gaussian_NB,
        Bernoulli_NB,
        Multinomial_NB,
        DecisionTree,
        RandomForest,
        ExtraTrees,
        AdaBoost,
        GradientBoosting,
        QDA,
        KNeighbors,
        Linear_SVC,
        RBF_SVM,
        GaussianProcess,
        XGBoost,
        LightGBM,
    ]

    def __init__(self):
        pass
