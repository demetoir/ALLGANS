from sklearn_like_toolkit.Base import BaseClass
from util.Logger import Logger
from util.numpy_utils import NP_ARRAY_TYPE_INDEX, reformat_np_arr
import numpy as np


class VotingClassifier(BaseClass):
    model_Ys_type = NP_ARRAY_TYPE_INDEX

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return super().__str__()

    def __init__(self, clfs, voting='hard', n_jobs=1):
        self.log = Logger(self.__class__.__name__)
        from sklearn.ensemble.voting_classifier import VotingClassifier

        if voting is 'soft':
            new_clfs = []
            for k, v in clfs:
                if hasattr(v, 'predict_proba'):
                    new_clfs += [(k, v)]
                else:
                    self.log.info(f'drop clf {k}')
            clfs = new_clfs

        self.clfs = clfs

        self.model = VotingClassifier(clfs, voting=voting, n_jobs=n_jobs)
        del VotingClassifier

    def fit(self, Xs, Ys, Ys_type=None):
        self.model.fit(Xs, reformat_np_arr(Ys, self.model_Ys_type, from_np_arr_type=Ys_type))

    def predict(self, Xs):
        return self.model.predict(Xs)

    def score(self, Xs, Ys, Ys_type=None):
        return self.model.score(Xs, reformat_np_arr(Ys, self.model_Ys_type, from_np_arr_type=Ys_type))

    def proba(self, Xs, transpose_shape=False):
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
