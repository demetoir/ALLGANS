from data_handler.BaseDataset import BaseDataset
from util.numpy_utils import NP_ARRAY_TYPE_INDEX, reformat_np_arr
from sklearn.ensemble.voting_classifier import VotingClassifier as _skVotingClassifier
from sklearn.ensemble import BaggingClassifier as _BaggingClassifier


class Dataset(BaseDataset):

    def load(self, path, limit=None):
        pass

    def save(self):
        pass

    def preprocess(self):
        pass


class skVoting(_skVotingClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX

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
