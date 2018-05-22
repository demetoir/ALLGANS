from sklearn_like_toolkit.sklearn_wrapper import BaseSklearnClassifier
from util.numpy_utils import NP_ARRAY_TYPE_INDEX


class XGBoost(BaseSklearnClassifier):
    model_Ys_type = NP_ARRAY_TYPE_INDEX
    tuning_grid = {
        'max_depth': [4, 6, 8],
        # 'n_estimators': [128, 256],
        # 'min_child_weight': [1, 2, 3],
        'gamma': [i / 10.0 for i in range(2, 10 + 1, 2)],
        'subsample': [i / 10.0 for i in range(2, 10 + 1, 2)],
        'colsample_bytree': [i / 10.0 for i in range(2, 10 + 1, 2)],
        # 'learning_rate': [0.01, 0.1, 1],
    }
    tuning_params = {
        'max_depth': 3,
        'n_estimators': 100,
        'min_child_weight': 1,
        'gamma': 0,

        'subsample': 1,
        'colsample_bytree': 1,
        'learning_rate': 0.1,
    }
    param = {
        'silent': True,
        'objective': 'binary:logistic',
        'booster': ['gbtree', 'gblinear', 'dart'],
        'colsample_bylevel': 1,

        'reg_alpha': 0,
        'reg_lambda': 1,

        'scale_pos_weight': 1,
        'max_delta_step': 0,

        'base_score': 0.5,
        'n_jobs': 1,
        'nthread': None,
        'random_state': 0,
        'seed': None,
        'missing': None,
    }

    def __init__(self, **params):
        super().__init__(**params)
        import warnings
        warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
        # params.update({"tree_method": 'gpu_hist'})
        # params.update({"tree_method": 'hist'})
        params.update({"tree_method": 'auto'})
        # params.update({"tree_method": 'exact'})
        # params.update({"tree_method": 'gpu_exact'})

        # params.update({'nthread': 1})
        # params.update({"silent": 1})
        import xgboost as xgb
        self.model = xgb.XGBClassifier(**params)
        del xgb
