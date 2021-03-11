# class definition imports
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin, TransformerMixin
from abc import ABCMeta, abstractmethod
import six

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

# required imports
import numpy as np


# General Class definitons
class TestPartition(six.with_metaclass(ABCMeta)):

    def __init__(self):
        pass

    def fit(self, X):
        self.X = X
        return self

    def test_power(self, x=None, p=2):
        if x is None:
            return np.power(self.X, p)
        else:
            return np.power(x, p)


def coeffofvar(df):

    col, cvar, na_perc = [], [], []
    for col_ in df.columns:
        if df[col_].dtype in [np.nan]:
            col.append(col_)
            val = df[col_].values
            cvar_ = np.nanstd(val) / np.nanmean(val)
            cvar.append(cvar_)
            na_prec_ = np.sum(np.isnan(val)) / val.shape[0]
            na_perc.append(na_prec_)
            print('Column: {} Coeff OF Var: {} NA perc: {}'.format(col_, cvar_, na_prec_))
    return col, cvar, na_perc


