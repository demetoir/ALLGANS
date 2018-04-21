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


class BaseSklearnClassfier:
    model_Ys_type = None

    def __str__(self):
        return self.__class__.__name__

    def __init__(self, *args, **kwargs):
        self.model = None

    def reformat_Ys(self, Ys, Ys_type=None):
        def get_Ys_type(Ys):
            shape = Ys.shape
            if len(shape) == 2:
                _type = "index"
            elif len(shape) == 3:
                _type = "onehot"
            else:
                _type = "invalid"

            return _type

        if Ys_type is None:
            Ys_type = get_Ys_type(Ys)

        convert_type = (Ys_type, self.model_Ys_type)

        if convert_type == ONEHOT_TO_INDEX:
            Ys = np_onehot_to_index(Ys)
        elif convert_type == INDEX_TO_ONEHOT:
            Ys = np_index_to_onehot(Ys)
        elif convert_type in NO_CONVERT:
            pass
        else:
            raise TypeError("Ys convert type Error Ys_type=%s, model_Ys_type=%s" % (Ys_type, self.model_Ys_type))

        return Ys

    def fit(self, Xs, Ys, Ys_type=None):
        self.model.fit(Xs, self.reformat_Ys(Ys, Ys_type=Ys_type))

    def predict(self, Xs):
        return self.model.predict(Xs)

    def acc(self, Xs, Ys, Ys_type=True):
        return self.model.score(Xs, self.reformat_Ys(Ys, Ys_type=Ys_type))

    def prob(self, Xs, transpose_shape=True):
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


class DecisionTree(BaseSklearnClassfier):
    """
    sklearn base DecisionTreeClassifier
    """
    model_Ys_type = YS_TYPE_ONEHOT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sklearn.tree import DecisionTreeClassifier as _d_tree
        self.model = _d_tree(*args, **kwargs)
        del _d_tree


class MLP(BaseSklearnClassfier):
    model_Ys_type = YS_TYPE_ONEHOT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sklearn.neural_network import MLPClassifier as _MLP
        self.model = _MLP(alpha=1)
        del _MLP


class Gaussian_NB(BaseSklearnClassfier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sklearn.naive_bayes import GaussianNB as _GaussianNB
        self.model = _GaussianNB()
        del _GaussianNB


class Bernoulli_NB(BaseSklearnClassfier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sklearn.naive_bayes import BernoulliNB as _BernoulliNB
        self.model = _BernoulliNB()
        del _BernoulliNB


class Multinomial_NB(BaseSklearnClassfier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sklearn.naive_bayes import MultinomialNB as _MultinomialNB
        self.model = _MultinomialNB()
        del _MultinomialNB


class QDA(BaseSklearnClassfier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as _QDA
        self.model = _QDA()
        del _QDA


class RandomForest(BaseSklearnClassfier):
    model_Ys_type = YS_TYPE_ONEHOT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier
        self.model = _RandomForestClassifier(
            max_depth=5,
            n_estimators=10,
            max_features=1
        )
        del _RandomForestClassifier


class AdaBoost(BaseSklearnClassfier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sklearn.ensemble import AdaBoostClassifier as _AdaBoostClassifier
        self.model = _AdaBoostClassifier()
        del _AdaBoostClassifier


class KNeighbors(BaseSklearnClassfier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier
        self.model = sklearn.neighbors.KNeighborsClassifier()
        del _KNeighborsClassifier


class Linear_SVM(BaseSklearnClassfier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sklearn.svm import SVC as _SVC
        self.model = _SVC(kernel="linear", C=0.025)
        del _SVC

    def prob(self, Xs, transpose_shape=True):
        print("""linear_SVM does not have attr "prob" """)
        return None


class RBF_SVM(BaseSklearnClassfier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sklearn.svm import SVC as _SVC
        self.model = _SVC(gamma=2, C=1)
        del _SVC

    def prob(self, Xs, transpose_shape=True):
        print("""linear_SVM does not have attr "prob" """)
        return None


class GaussianProcess(BaseSklearnClassfier):
    model_Ys_type = YS_TYPE_INDEX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sklearn.gaussian_process import GaussianProcessClassifier as _GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF as _RBF
        self.model = _GaussianProcessClassifier(1.0 * _RBF(1.0))
        del _RBF
        del _GaussianProcessClassifier


classifiers = [
    Gaussian_NB,
    Bernoulli_NB,
    Multinomial_NB,
    DecisionTree,
    RandomForest,
    AdaBoost,
    MLP,
    QDA,
    KNeighbors,
    Linear_SVM,
    RBF_SVM,
    GaussianProcess,
]
