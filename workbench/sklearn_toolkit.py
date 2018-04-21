import numpy as np
import sklearn
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
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


class BaseSklearn:
    def __str__(self):
        return self.__class__.__name__

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

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.neural_network import MLPClassifier as _MLP
        self.model = _MLP(alpha=1)
        del _MLP


class Gaussian_NB(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.naive_bayes import GaussianNB as _GaussianNB
        self.model = _GaussianNB(**params)
        del _GaussianNB


class Bernoulli_NB(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.naive_bayes import BernoulliNB as _BernoulliNB
        self.model = _BernoulliNB(**params)
        del _BernoulliNB


class Multinomial_NB(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.naive_bayes import MultinomialNB as _MultinomialNB
        self.model = _MultinomialNB(**params)
        del _MultinomialNB


class QDA(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX

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

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.tree import DecisionTreeClassifier as _DecisionTreeClassifier
        self.model = _DecisionTreeClassifier(**params)
        del _DecisionTreeClassifier


class RandomForest(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_ONEHOT

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier
        self.model = _RandomForestClassifier(**params)
        del _RandomForestClassifier


class ExtraTrees(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_ONEHOT

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.ensemble import ExtraTreesClassifier as _ExtraTreesClassifier
        self.model = _ExtraTreesClassifier(**params)
        del _ExtraTreesClassifier


class AdaBoost(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.ensemble import AdaBoostClassifier as _AdaBoostClassifier
        self.model = _AdaBoostClassifier(**params)
        del _AdaBoostClassifier


class GradientBoosting(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.ensemble import GradientBoostingClassifier as _GradientBoostingClassifier
        self.model = _GradientBoostingClassifier(**params)
        del _GradientBoostingClassifier


class KNeighbors(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier
        self.model = sklearn.neighbors.KNeighborsClassifier(**params)
        del _KNeighborsClassifier


class Linear_SVM(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.svm import SVC as _SVC
        # TODO **params
        self.model = _SVC(kernel="linear", C=0.025)
        del _SVC

    def proba(self, Xs, transpose_shape=True):
        print("""linear_SVM does not have attr "prob" """)
        return None


class RBF_SVM(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX

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

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.gaussian_process import GaussianProcessClassifier as _GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF as _RBF
        # TODO **params
        self.model = _GaussianProcessClassifier(1.0 * _RBF(1.0))
        del _RBF
        del _GaussianProcessClassifier


class SGD(BaseSklearnClassifier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.linear_model import SGDClassifier as _SGDClassifier
        self.model = _SGDClassifier(**params)
        del _SGDClassifier

    def proba(self, Xs, transpose_shape=True):
        # todo
        return None
        # return super().proba(Xs, transpose_shape)

