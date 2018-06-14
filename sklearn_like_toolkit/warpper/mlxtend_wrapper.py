import warnings

from util.numpy_utils import NP_ARRAY_TYPE_INDEX, reformat_np_arr
from mlxtend.classifier import Adaline as _Adaline
from mlxtend.classifier import EnsembleVoteClassifier as _EnsembleVoteClassifier
from mlxtend.classifier import LogisticRegression as _LogisticRegression
from mlxtend.classifier import MultiLayerPerceptron as _MultiLayerPerceptron
from mlxtend.classifier import Perceptron as _Perceptron
from mlxtend.classifier import SoftmaxRegression as _SoftmaxRegression
from mlxtend.classifier import StackingCVClassifier as _StackingCVClassifier
from mlxtend.classifier import StackingClassifier as _StackingClassifier
from sklearn_like_toolkit.base._base import _clf_metric, _Reformat_Ys


class mlxAdalineClf(_Adaline):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
    tuning_grid = {
        'eta': [i / 10.0 for i in range(1, 10 + 1, 3)],
        'epochs': [64, 128, 256],
        'minibatches': [1, 2, 4, 8],
    }
    tuning_params = {
    }
    remain_param = {
    }

    def __init__(self, eta=0.01, epochs=50, minibatches=None, random_seed=None, print_progress=0):
        minibatches = 1
        super().__init__(eta, epochs, minibatches, random_seed, print_progress)

    def fit(self, X, y, init_params=True):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, init_params)

    def score(self, X, y):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y)


class mlxLogisticRegressionClf(_LogisticRegression):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
    tuning_grid = {
        'eta': [i / 10.0 for i in range(1, 10 + 1, 3)],
        'epochs': [64, 128, 256],
        'minibatches': [1, 2, 4, 8],
        'l2_lambda': [i / 10.0 for i in range(1, 10 + 1, 3)]
    }
    tuning_params = {
    }
    remain_param = {
    }

    def __init__(self, eta=0.01, epochs=50, l2_lambda=0.0, minibatches=1, random_seed=None, print_progress=0):
        super().__init__(eta, epochs, l2_lambda, minibatches, random_seed, print_progress)

    def fit(self, X, y, init_params=True):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, init_params)

    def score(self, X, y):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y)


class mlxMLPClf(_MultiLayerPerceptron):
    # todo add param grid
    model_Ys_type = NP_ARRAY_TYPE_INDEX
    tuning_grid = {
        'eta': [i / 10.0 for i in range(1, 10 + 1, 3)],
        'epochs': [64, 128, 256],
    }
    tuning_params = {
    }
    remain_param = {
    }

    def __init__(self, eta=0.5, epochs=50, hidden_layers=None, n_classes=None, momentum=0.0, l1=0.0, l2=0.0,
                 dropout=1.0, decrease_const=0.0, minibatches=1, random_seed=None, print_progress=0):
        warnings.filterwarnings(module='mlxtend*', action='ignore', category=FutureWarning)
        warnings.filterwarnings(module='mlxtend*', action='ignore', category=RuntimeWarning)
        if hidden_layers is None:
            hidden_layers = [50]
        super().__init__(eta, epochs, hidden_layers, n_classes, momentum, l1, l2, dropout, decrease_const, minibatches,
                         random_seed, print_progress)

    def fit(self, X, y, init_params=True):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, init_params)

    def score(self, X, y):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y)


class mlxPerceptronClf(_Perceptron):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
    tuning_grid = {
        'eta': [i / 10.0 for i in range(1, 10 + 1, 3)],
        'epochs': [64, 128, 256],
    }
    tuning_params = {
    }
    remain_param = {
    }

    def __init__(self, eta=0.1, epochs=50, random_seed=None, print_progress=0):
        super().__init__(eta, epochs, random_seed, print_progress)

    def fit(self, X, y, init_params=True):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, init_params)

    def score(self, X, y):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y)


class mlxSoftmaxRegressionClf(_SoftmaxRegression):
    model_Ys_type = NP_ARRAY_TYPE_INDEX

    tuning_grid = {
        'eta': [i / 10.0 for i in range(1, 10 + 1, 3)],
        'epochs': [64, 128, 256],
        'l2': [i / 10.0 for i in range(1, 10 + 1, 3)],
        'minibatches': [1, 2, 4],
    }
    tuning_params = {
    }
    remain_param = {
    }

    def __init__(self, eta=0.01, epochs=50, l2=0.0, minibatches=1, n_classes=None, random_seed=None, print_progress=0):
        super().__init__(eta, epochs, l2, minibatches, n_classes, random_seed, print_progress)

    def fit(self, X, y, init_params=True):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, init_params)

    def score(self, X, y):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y)


class mlxVotingClf(_EnsembleVoteClassifier):
    # todo add param grid
    model_Ys_type = NP_ARRAY_TYPE_INDEX

    def __init__(self, clfs, voting='hard', weights=None, verbose=0, refit=True):
        super().__init__(clfs, voting, weights, verbose, refit)

    def fit(self, X, y):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y)

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)


class mlxStackingClf(_StackingClassifier, _clf_metric, _Reformat_Ys):
    # todo add param grid
    model_Ys_type = NP_ARRAY_TYPE_INDEX

    def __init__(self, classifiers, meta_classifier, use_probas=False, average_probas=False, verbose=0,
                 use_features_in_secondary=False, store_train_meta_features=False, use_clones=True):
        _StackingClassifier.__init__(self, classifiers, meta_classifier, use_probas, average_probas, verbose,
                                     use_features_in_secondary,
                                     store_train_meta_features, use_clones)
        _clf_metric.__init__(self)

    def fit(self, X, y):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y)

    def score(self, X, y, metric='accuracy'):
        y = reformat_np_arr(y, self.model_Ys_type)
        return self._apply_metric(y, self.predict(X), metric)

    def score_pack(self, X, y):
        y = reformat_np_arr(y, self.model_Ys_type)
        return self._apply_metric_pack(y, self.predict(X))


class mlxStackingCVClf(_StackingCVClassifier, _clf_metric, _Reformat_Ys):
    # todo add param grid
    model_Ys_type = NP_ARRAY_TYPE_INDEX

    def __init__(self, classifiers, meta_classifier, use_probas=False, cv=2, use_features_in_secondary=False,
                 stratify=True, shuffle=True, verbose=0, store_train_meta_features=False, use_clones=True):
        super().__init__(classifiers, meta_classifier, use_probas, cv, use_features_in_secondary, stratify, shuffle,
                         verbose, store_train_meta_features, use_clones)

    def fit(self, X, y, groups=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().fit(X, y, groups)

    def score(self, X, y, metric='accuracy'):
        y = reformat_np_arr(y, self.model_Ys_type)
        return self._apply_metric(y, self.predict(X), metric)

    def score_pack(self, X, y):
        y = reformat_np_arr(y, self.model_Ys_type)
        return self._apply_metric_pack(y, self.predict(X))
