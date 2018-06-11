from util.numpy_utils import NP_ARRAY_TYPE_INDEX, reformat_np_arr
import warnings
import lightgbm
import numpy as np


class LightGBMClf(lightgbm.LGBMClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
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
    remain_param = {
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

    def __init__(self, boosting_type="gbdt", num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100,
                 subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0., min_child_weight=1e-3,
                 min_child_samples=20, subsample=1., subsample_freq=1, colsample_bytree=1., reg_alpha=0., reg_lambda=0.,
                 random_state=None, n_jobs=-1, silent=True, **kwargs):
        kwargs['verbose'] = -1
        warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
        super().__init__(boosting_type, num_leaves, max_depth, learning_rate, n_estimators, subsample_for_bin,
                         objective, class_weight, min_split_gain, min_child_weight, min_child_samples, subsample,
                         subsample_freq, colsample_bytree, reg_alpha, reg_lambda, random_state, n_jobs, silent,
                         **kwargs)

    def fit(self, X, y, sample_weight=None, init_score=None, eval_set=None, eval_names=None, eval_sample_weight=None,
            eval_class_weight=None, eval_init_score=None, eval_metric="logloss", early_stopping_rounds=None,
            verbose=True, feature_name='auto', categorical_feature='auto', callbacks=None):
        y = reformat_np_arr(y, self.model_Ys_type)

        return super().fit(X, y, sample_weight, init_score, eval_set, eval_names, eval_sample_weight, eval_class_weight,
                           eval_init_score, eval_metric, early_stopping_rounds, verbose, feature_name,
                           categorical_feature, callbacks)

    def predict_proba(self, X, raw_score=False, num_iteration=0, transpose_shape=False):
        ret = super().predict_proba(X, raw_score, num_iteration)

        if transpose_shape is True:
            ret = np.transpose(ret, axes=(1, 0, 2))

        return ret

    def score(self, X, y, sample_weight=None):
        y = reformat_np_arr(y, self.model_Ys_type)
        return super().score(X, y, sample_weight)
