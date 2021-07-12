from classifiers.base_classifier import BaseClassifier
import scipy.stats as stats
import xgboost as xgb

class XGBoostClassifier(BaseClassifier):
    def __init__(self):
        super().__init__('XGBoost')
        tree_method = "hist"
        self.classifier: xgb.XGBClassifier = xgb.XGBClassifier(tree_method=tree_method, n_estimators=600, colsample_bytree=0.8,eta=0.2, eval_metric='mlogloss')
        self.use_parameter_search: bool = True
        self.search_parameters = [
            {"n_estimators": stats.randint(100, 1200), "colsample_bytree": [1, 0.9, 0.8, 0.5],
             "eta": stats.expon(scale=.2), "max_depth": stats.randint(2, 12), "gamma": [0, 2, 4],
             # "lambda": stats.uniform(0.0, 2.0), "alpha": stats.uniform(0.0, 2.0),
             "min_child_weight": stats.randint(1, 3)}]
