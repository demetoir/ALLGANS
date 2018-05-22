from sklearn.gaussian_process.kernels import RBF
from sklearn_like_toolkit.base import BaseSklearn
import numpy as np
import sklearn

from util.numpy_utils import reformat_np_arr, NP_ARRAY_TYPE_ONEHOT, NP_ARRAY_TYPE_INDEX


class BaseSklearnClassifier(BaseSklearn):
    model_Ys_type = None
    tuning_params = None
    tuning_grid = None

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return super().__str__()

    def __init__(self, **params):
        super().__init__(**params)
        self.model = None

    def fit(self, Xs, Ys, Ys_type=None):
        self.model.fit(Xs, reformat_np_arr(Ys, self.model_Ys_type, from_np_arr_type=Ys_type))

    def predict(self, Xs):
        return self.model.predict(Xs)

    def score(self, Xs, Ys, Ys_type=None):
        return self.model.score(Xs, reformat_np_arr(Ys, self.model_Ys_type, from_np_arr_type=Ys_type))

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


class skMLP(BaseSklearnClassifier):
    model_Ys_type = NP_ARRAY_TYPE_ONEHOT
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


class skGaussian_NB(BaseSklearnClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
    tuning_grid = {}
    tuning_params = {
        'priors': None
    }

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.naive_bayes import GaussianNB as _GaussianNB
        self.model = _GaussianNB(**params)
        del _GaussianNB


class skBernoulli_NB(BaseSklearnClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
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


class skMultinomial_NB(BaseSklearnClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
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


class skQDA(BaseSklearnClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
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


class skDecisionTree(BaseSklearnClassifier):
    """
    sklearn base DecisionTreeClassifier
    """
    model_Ys_type = NP_ARRAY_TYPE_ONEHOT
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


class skRandomForest(BaseSklearnClassifier):
    model_Ys_type = NP_ARRAY_TYPE_ONEHOT
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


class skExtraTrees(BaseSklearnClassifier):
    model_Ys_type = NP_ARRAY_TYPE_ONEHOT
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


class skAdaBoost(BaseSklearnClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
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


class skGradientBoosting(BaseSklearnClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
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


class skKNeighbors(BaseSklearnClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
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


class skLinear_SVC(BaseSklearnClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
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


class skRBF_SVM(BaseSklearnClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
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


class skGaussianProcess(BaseSklearnClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
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


class skSGD(BaseSklearnClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX

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


