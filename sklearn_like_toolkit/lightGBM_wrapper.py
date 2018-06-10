import warnings

from sklearn_like_toolkit.sklearn_wrapper import BaseSklearnClassifier
from util.numpy_utils import NP_ARRAY_TYPE_INDEX


class LightGBM(BaseSklearnClassifier):
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
        warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

        import lightgbm as lgbm
        params.update(self.param)
        self.model = lgbm.LGBMClassifier(**params)
        del lgbm

    def predict_proba(self, Xs):
        return self.proba(Xs, transpose_shape=False )
