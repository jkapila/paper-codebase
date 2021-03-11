from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

import numpy as np
import pandas as pd

class ExtendedMinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self, feature_range=(-1, 1), na_treatment='replace', na_value=-1, treat_inf_as_na=True,
                 data_min=None, data_max=None, copy=False, verbose=False):
        self.feature_range = feature_range
        self.copy = copy
        self.verbose = verbose
        self.na_treatment = na_treatment
        self.na_value = na_value
        self.treat_inf_as_na = treat_inf_as_na

        if data_max is not None and isinstance(data_max, pd.DataFrame):
            self.data_max = data_max.values
        elif data_max is not None and isinstance(data_max, np.ndarray):
            self.data_max = data_max
        else:
            print("Max values not in correct format!")
            self.data_max = None

        if data_min is not None and isinstance(data_min, pd.DataFrame):
            self.data_min = data_min.values
        elif data_min is not None and isinstance(data_min, np.ndarray):
            self.data_min = data_min
        else:
            print("Min values not in correct format!")
            self.data_min = None

    def fit(self, X):
        """Compute the minimum and maximum to be used for later scaling.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        """
        X = check_array(X, copy=self.copy, ensure_2d=False, force_all_finite=False)
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(feature_range))

        if self.data_min is not None:
            assert len(self.data_min) == X.shape[1]
            data_min = self.data_min
        else:
            data_min = np.nanmin(X, axis=0)
            self.data_min = data_min

        if self.data_max is not None:
            assert len(self.data_max) == X.shape[1]
            data_max = self.data_max
        else:
            data_max = np.nanmax(X, axis=0)
            self.data_max = data_max

        if self.treat_inf_as_na:
            X[np.isinf(X)] = np.nan

        if self.na_treatment == 'max':
            self.na_treatment_value = data_max
        elif self.na_treatment == 'min':
            self.na_treatment_value = data_min
        elif self.na_treatment == 'max_perc':
            self.na_treatment_value = data_max * (1 + self.na_value)
        elif self.na_treatment == 'min_perc':
            self.na_treatment_value = data_min * (1 - self.na_value)
        elif self.na_treatment == 'replace':
            self.na_treatment_value = self.na_value
        else:  # default behaviour mid value of range
            self.na_treatment_value = (data_max - data_min) / 2

        data_range = data_max - data_min

        if self.verbose:
            print('Minmum Values: \n{}'.format(data_min))
            print('Maximum Values: \n{}'.format(data_max))
            print('Data_range: \n{}'.format(data_range))
            print('NA treatment values: \n{}'.format(self.na_treatment_value))

        # Do not scale constant features
        if isinstance(data_range, np.ndarray):
            data_range[data_range == 0.0] = 1.0
        elif data_range == 0.:
            data_range = 1.
        self.scale_ = (feature_range[1] - feature_range[0]) / data_range
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_range = data_range

        return self

    def transform(self, X, do_transform=True):
        """Scaling features of X according to feature_range.
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            Input data that will be transformed.
        """
        check_is_fitted(self, 'scale_')

        X = check_array(X, copy=self.copy, ensure_2d=False, force_all_finite=False)

        if self.treat_inf_as_na:
            X[np.isinf(X)] = np.nan

        mask = np.isnan(X)
        if X.shape[0] > 1:
            na_values = self.na_treatment_value * np.ones((X.shape[0], 1))
            # print(X.shape,na_values.shape)
            assert X.shape == na_values.shape
        else:
            na_values = self.na_treatment_value
            print(X.shape, na_values.shape)

        X[mask] = na_values[mask]
        if do_transform:
            X *= self.scale_
            X += self.min_

        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            Input data that will be transformed.
        """
        check_is_fitted(self, 'scale_')

        X = check_array(X, copy=self.copy, ensure_2d=False, force_all_finite=False)
        if self.treat_inf_as_na:
            X[np.isinf(X)] = np.nan
        X -= self.min_
        X /= self.scale_
        return X
