from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from . import config

class ModelFactory:
    @staticmethod
    def get_lgbm():
        return LGBMClassifier(**config.LGBM_PARAMS)

    @staticmethod
    def get_catboost(cat_features=None):
        params = config.CATBOOST_PARAMS.copy()
        if cat_features:
            params['cat_features'] = cat_features
        return CatBoostClassifier(**params)

    @staticmethod
    def get_xgboost():
        return XGBClassifier(**config.XGB_PARAMS)

    @staticmethod
    def get_svm():
        return SVC(**config.SVM_PARAMS)

    @staticmethod
    def get_nn():
        return MLPClassifier(**config.NN_PARAMS)

    @staticmethod
    def get_meta_learner():
        if config.META_MODEL_TYPE == 'random_forest':
            return RandomForestClassifier(**config.META_RF_PARAMS)
        return LogisticRegression(**config.LOGISTIC_PARAMS)
