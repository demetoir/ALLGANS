import warnings

from sklearn.gaussian_process.kernels import RBF as _RBF
from sklearn.linear_model.stochastic_gradient import DEFAULT_EPSILON
from sklearn.neural_network import MLPClassifier as _skMLPClassifier
from sklearn.naive_bayes import GaussianNB as _skGaussianNB
from sklearn.naive_bayes import BernoulliNB as _skBernoulliNB
from sklearn.naive_bayes import MultinomialNB as _skMultinomialNB
from sklearn.linear_model import SGDClassifier as _skSGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier as _skGaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier as _skKNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier as _skGradientBoostingClassifier, \
    VotingClassifier as _skVotingClassifier, BaggingClassifier as _BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier as _skAdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier as _skExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier as _skRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier as _skDecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as _skQDA
from sklearn.svm import LinearSVC as _skLinearSVC
from sklearn.svm import SVC as _skSVC
from util.numpy_utils import reformat_np_arr, NP_ARRAY_TYPE_INDEX


class skMLP(_skMLPClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
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

    def fit(self, X, y):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)


class skGaussian_NB(_skGaussianNB):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
    tuning_grid = {}
    tuning_params = {
        'priors': None
    }

    def fit(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, sample_weight)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)


class skBernoulli_NB(_skBernoulliNB):
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

    def fit(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, sample_weight)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)


class skMultinomial_NB(_skMultinomialNB):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
    tuning_grid = {
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    }
    tuning_params = {
        'alpha': 1.0,
        'class_prior': None,
        'fit_prior': True
    }

    def fit(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, sample_weight)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)


class skQDA(_skQDA):
    def __init__(self, priors=None, reg_param=0., store_covariance=False, tol=1.0e-4, store_covariances=None):
        warnings.filterwarnings(module='sklearn*', action='ignore', category=Warning)
        super().__init__(priors, reg_param, store_covariance, tol, store_covariances)

    model_Ys_type = NP_ARRAY_TYPE_INDEX
    tuning_grid = {

    }
    tuning_params = {

    }
    remain_param = {
        # TODO
        # ? ..
        'priors': None,
        'reg_param': 0.0,
        'store_covariance': False,
        'store_covariances': None,
        'tol': 0.0001
    }

    def fit(self, X, y):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)


class skDecisionTree(_skDecisionTreeClassifier):
    """
    sklearn base DecisionTreeClassifier
    """
    model_Ys_type = NP_ARRAY_TYPE_INDEX
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

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, sample_weight, check_input, X_idx_sorted)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)


class skRandomForest(_skRandomForestClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
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

    def fit(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, sample_weight)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)


class skExtraTrees(_skExtraTreesClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
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

    def fit(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, sample_weight)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)


class skAdaBoost(_skAdaBoostClassifier):
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

    def fit(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, sample_weight)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)


class skGradientBoosting(_skGradientBoostingClassifier):
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

    def fit(self, X, y, sample_weight=None, monitor=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, sample_weight, monitor)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)

    def _make_estimator(self, append=True):
        pass


class skKNeighbors(_skKNeighborsClassifier):
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

    def fit(self, X, y):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)


class skGaussianProcess(_skGaussianProcessClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
    # todo
    tuning_grid = {

    }
    tuning_params = {

    }
    remain_param = {
        'kernel': 1 ** 2 * _RBF(length_scale=1),
        'kernel__k1': 1 ** 2,
        'kernel__k1__constant_value': 1.0,
        'kernel__k1__constant_value_bounds': (1e-05, 100000.0),
        'kernel__k2': _RBF(length_scale=1),
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

    def fit(self, X, y):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y)

    def predict_proba(self, X):
        return super().predict_proba(X)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)


class skSGD(_skSGDClassifier):
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
    remain_param = {
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

    def __init__(self, loss="hinge", penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000,
                 tol=None, shuffle=True, verbose=0, epsilon=DEFAULT_EPSILON, n_jobs=1, random_state=None,
                 learning_rate="optimal", eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False,
                 n_iter=None):
        super().__init__(loss, penalty, alpha, l1_ratio, fit_intercept, max_iter, tol, shuffle, verbose, epsilon,
                         n_jobs, random_state, learning_rate, eta0, power_t, class_weight, warm_start, average, n_iter)

    def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, coef_init, intercept_init, sample_weight)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)


class skLinear_SVC(_skLinearSVC):
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

    @property
    def predict_proba(self):
        return self._predict_proba_lr

    def fit(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, sample_weight)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)


class skRBF_SVM(_skSVC):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True,
                 tol=1e-3, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):
        super().__init__(C, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight,
                         verbose, max_iter, decision_function_shape, random_state)

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
    remain_param = {
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

    def fit(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, sample_weight)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)


class skVoting(_skVotingClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
    tuning_grid = {
    }
    tuning_params = {
    }

    def __init__(self, estimators, voting='hard', weights=None, n_jobs=1, flatten_transform=None):
        super().__init__(estimators, voting, weights, n_jobs, flatten_transform)

    def fit(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, sample_weight)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)


class skBagging(_BaggingClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
    tuning_grid = {
    }
    tuning_params = {
    }

    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True,
                 bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0):
        super().__init__(base_estimator, n_estimators, max_samples, max_features, bootstrap, bootstrap_features,
                         oob_score, warm_start, n_jobs, random_state, verbose)

    def fit(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, sample_weight)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)
