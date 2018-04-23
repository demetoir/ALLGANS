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

    def __init__(self, estimator, param_grid, **kwargs):
        super().__init__(estimator, param_grid, **kwargs)
        self.estimator = estimator
        self.param_grid = param_grid

        from sklearn.model_selection import GridSearchCV as _GridSearchCV
        self.param_search = _GridSearchCV(estimator, param_grid, **kwargs)
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
        super().get_params(deep)
        return self.param_search.get_params(deep=deep)

    def set_params(self, **params):
        super().set_params(**params)
        return self.param_search.set_params(**params)


class BaseSklearnClassifier(BaseSklearn):
    model_Ys_type = None

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

    param = {
        'hidden_layer_sizes': (100,),
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'max_iter': 200,
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'learning_rate_init': 0.001,
        'early_stopping': False,
        'tol': 0.0001,

        # adam solver option
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-08,

        # sgd solver option
        # between 0 and 1.
        'momentum': 0.9,
        'nesterovs_momentum': True,
        'power_t': 0.5,

        # L2 penalty
        'alpha': 1,

        # batch option
        'batch_size': 'auto',
        'shuffle': True,
        'validation_fraction': 0.1,

        # etc
        'random_state': None,
        'verbose': False,
        'warm_start': False
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.neural_network import MLPClassifier as _MLP
        self.model = _MLP(alpha=1)
        del _MLP


class Gaussian_NB(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX

    param = {
        # ex [0.3, 0.7]
        'priors': None
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.naive_bayes import GaussianNB as _GaussianNB
        self.model = _GaussianNB(**params)
        del _GaussianNB


class Bernoulli_NB(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX
    param = {
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

    param = {
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
    param = {
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
    param = {
        # tuning params
        'max_depth': None,
        'min_samples_leaf': 1,
        'min_samples_split': 2,

        # class weight options
        'class_weight': None,
        'min_weight_fraction_leaf': 0.0,

        # etc
        'presort': False,
        'random_state': None,

        # only use default option
        'criterion': 'gini',
        'splitter': 'best',
        'max_leaf_nodes': None,
        'max_features': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,

    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.tree import DecisionTreeClassifier as _DecisionTreeClassifier
        self.model = _DecisionTreeClassifier(**params)
        del _DecisionTreeClassifier


class RandomForest(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_ONEHOT
    param = {
        # tuning params
        'n_estimators': 10,
        'max_depth': None,
        'min_samples_leaf': 1,
        'min_samples_split': 2,

        # class weight option
        'class_weight': None,
        'min_weight_fraction_leaf': 0.0,

        # etc
        'n_jobs': 1,
        'oob_score': False,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,

        # only use default option
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

    param = {
        # tuning params
        'n_estimators': 10,
        'max_depth': None,
        'min_samples_leaf': 1,
        'min_samples_split': 2,

        # class weight options
        'class_weight': None,
        'min_weight_fraction_leaf': 0.0,

        # etc
        'n_jobs': 1,
        'oob_score': False,
        'random_state': None,
        'verbose': 0,
        'warm_start': False,

        # only use default
        'bootstrap': False,
        'criterion': 'gini',
        'max_features': 'auto',
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,

    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.ensemble import ExtraTreesClassifier as _ExtraTreesClassifier
        self.model = _ExtraTreesClassifier(**params)
        del _ExtraTreesClassifier


class AdaBoost(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX

    param = {
        'base_estimator': None,

        # tuning param
        'learning_rate': 1.0,
        'n_estimators': 50,

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

    param = {
        # tuning param
        'learning_rate': 0.1,
        'max_depth': 3,
        'n_estimators': 100,
        'min_samples_leaf': 1,
        'min_samples_split': 2,

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

        # use only default
        'criterion': 'friedman_mse',
        'min_impurity_decrease': 0.0,
        'min_impurity_split': None,
        'max_features': None,
        'max_leaf_nodes': None,
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.ensemble import GradientBoostingClassifier as _GradientBoostingClassifier
        self.model = _GradientBoostingClassifier(**params)
        del _GradientBoostingClassifier


class KNeighbors(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX
    param = {
        # tuning param
        'n_neighbors': 5,

        # use default ?
        'weights': 'uniform',
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': 2,
        'leaf_size': 30,

        # etc
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
    param = {
        # tuning param
        'C': 1.0,
        'max_iter': 1000,

        'class_weight': None,

        'dual': True,

        'multi_class': ['ovr', 'crammer_singer'],
        'loss': ['squared_hinge', 'hinge'],
        'penalty': ['l2', 'l1'],

        # use default
        'fit_intercept': True,
        'intercept_scaling': 1,

        # etc
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
    # todo
    param = {'C': 1,
             'cache_size': 200,
             'class_weight': None,
             'coef0': 0.0,
             'decision_function_shape': 'ovr',
             'degree': 3,
             'gamma': 2,
             'kernel': 'rbf',
             'max_iter': -1,
             'probability': False,
             'random_state': None,
             'shrinking': True,
             'tol': 0.001,
             'verbose': False}

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

    param = {
        'copy_X_train': True,
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
        'warm_start': False
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
    params = {
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


class SklearnClassifierPack(BaseClass):
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

    def __init__(self):
        pass
