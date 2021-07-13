from typing import List
from pandas import DataFrame
import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        is_na = None

        for column in self.columns:
            X.loc[~X[column].isna(), column] = 1
            X.loc[X[column].isna(), column] = 0

            if is_na is not None:
                is_na = is_na & X[column] == 0
            else:
                is_na = X[column] == 0

        return X


class ListTabSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: DataFrame):
        for column in self.columns:
            X[column] = X[column].str.split('\t')

        return X


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for column in self.columns:
            result = self.one_hot_encoder.fit_transform(X[[column]])
            columns = ['{}_{}'.format(column, i) for i in range(result.shape[1])]
            tmp_df = pd.DataFrame(result, columns=columns)

            X = pd.concat([X, tmp_df], axis=1, sort=False)
        return X


class ListCountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for column in self.columns:
            X['{}_count'.format(column)] = X[column].map(lambda x: len(x) if not isinstance(x, float) else 0)

        return X


class NumericQuantileBucketOneHotEncoder:
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        est = KBinsDiscretizer(encode='onehot-dense', strategy='quantile')
        for column in self.columns:
            result = est.fit_transform(X[[column]])
            columns = ['{}_{}'.format(column, i) for i in range(5)]

            tmp_df = pd.DataFrame(result, columns=columns)

            X = pd.concat([X, tmp_df], axis=1, sort=False)

        return X