from abc import ABCMeta, abstractmethod
import six
import os
import sys
import time

from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline, Parallel
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class ExtendedMinMaxScaler(BaseEstimator, TransformerMixin):


    def __init__(self, feature_range=(0, 1), na_treatment='replace', na_value=-1, treat_inf_as_na=True,
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

    def transform(self, X):
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
            #print(X.shape,na_values.shape)
            assert X.shape == na_values.shape
        else:
            na_values = self.na_treatment_value
            print(X.shape,na_values.shape)
            
        X[mask] = na_values[mask]
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


class OneClassLookalike(six.with_metaclass(ABCMeta)):

    def __init__(self, path, folder_name, limit=0.01, minmaxparams=None, oneclassparams=None, na_treatment="mode"):
        self.limit = limit
        self.minmaxparams = minmaxparams
        self.oneclassparams = oneclassparams
        self.continuous_stats = None
        self.categorical_stats = None
        self.na_treatment = na_treatment
        self.path = path
        self.folder_name = folder_name
        self.selected_fields = None
        self.selected_index = None
        self.input_length = None
        self.pipe = None
        self.processed_stat = None
        self.__get_stats()

    def __get_stats(self):
        stats = pd.read_csv(os.path.join(self.path, 'continuous.csv'))
        stats.columns = [i.lower().replace(' ', '_') for i in stats.columns]
        stats = stats[[i for i in stats.columns if not str(i).startswith('unnamed')]]
        # print(stats)
        stats = stats[stats.folder.isin([self.folder_name])]
        stats[['mean', 'std_dev', 'min', 'max', 'coeffvar']] = stats[['mean', 'std_dev', 'min',
                                                                      'max', 'coeffvar']].apply(pd.to_numeric,
                                                                                                errors='omit')
        # print(stats.isna().sum())
        self.continuous_stats = stats
        # print(self.continuous_stats.head(10))
        # print(self.continuous_stats.dtypes)
        # print(self.continuous_stats.columns)

    def __coeffofvar(self, df):

        col, cvar, na_perc = [], [], []
        for col_ in df.columns:
            if df[col_].dtypes in [np.nan]:
                col.append(col_)
                val = df[col_].values
                cvar_ = np.nanstd(val) / np.nanmean(val)
                cvar.append(cvar_)
                na_prec_ = np.sum(np.isnan(val)) / val.shape[0]
                na_perc.append(na_prec_)
                print('Column: {} Coeff OF Var: {} NA perc: {}'.format(col_, cvar_, na_prec_))
        opt = {'field_name_header': col, 'coeffvar_pos': cvar, 'na_percentage_pos': na_perc}
        return opt

    def fit(self, df):

        t = time.time()
        # if isinstance(df,pd.DataFrame):
        df.columns = [i.lower().replace(' ', '_') for i in df.columns]
        self.input_length = df.shape[1]
        cont_df = df[self.continuous_stats.field_name_header]
        cont_df = cont_df.apply(pd.to_numeric, errors='coerce')
        # cont_stat = self.__coeffofvar(cont_df)
        cont_stat = cont_df.describe().T
        # print(cont_stat)
        cont_stat['coeffvar_pos'] = cont_stat['std'] / cont_stat['mean']
        cont_stat['field_name_header'] = cont_stat.index
        cont_stat['na_perc'] = cont_df.isna().sum() / cont_df.shape[0]

        cont_stat = pd.merge(self.continuous_stats, cont_stat, how='inner', on='field_name_header')

        cont_stat['coeff_diff'] = cont_stat.coeffvar - cont_stat.coeffvar_pos
        cont_stat['sel_fields'] = cont_stat['coeff_diff'].apply(
            lambda x: 1 if x >= self.limit or x <= -self.limit else 0)
        # print(cont_stat.head().T)
        # print(np.sum(cont_stat['sel_fields']))
        self.processed_stat = cont_stat
        sel_fields = cont_stat.loc[cont_stat.sel_fields == 1, :]
        self.selected_fields = sel_fields.field_name_header.values.T.tolist()
        print('Total Fields Selected {}'.format(len(self.selected_fields)))
        self.selected_index = [i for i, v in enumerate(df.columns) if v in self.selected_fields]
        print('Finding relevant fields took {:5.4f} secs'.format(time.time() - t))

        t = time.time()
        minmaxer = ExtendedMinMaxScaler(na_treatment='max_perc', na_value=0.25,
                                        data_min=cont_stat.loc[cont_stat.sel_fields == 1, 'min_x'].values.T,
                                        data_max=cont_stat.loc[cont_stat.sel_fields == 1, 'max_x'].values.T
                                        ).fit(cont_df[self.selected_fields])
        cont_df = minmaxer.transform(cont_df[self.selected_fields])
        print('Scaling data done in {:5.4f} secs\nMaking Classifier'.format(time.time() - t))

        t = time.time()
        # clf = OneClassSVM().fit(cont_df)
        clf = IsolationForest(n_estimators=250, n_jobs=-1, max_features=50,
                              contamination=0.2, bootstrap=True).fit(cont_df)

        # clf = LocalOutlierFactor(n_jobs=-1,contamination=0.3).fit(cont_df)
        print('Building One Class model took {:5.4f} secs'.format(time.time() - t))
        self.scaler = minmaxer
        self.lookalike_scorer = clf
        # self.pipe = Pipeline([('minmaxer',minmaxer),('oneclasser', clf)])
        # self.pipe.fit(cont_df[self.selected_fields])
        return self

    def predict(self, X):
        
        # if isinstance(X, np.ndarray):
        #     if X.shape[1] == self.input_length:
        #         X_ = X[:, self.selected_index]
        #
        #     else:
        #         #assert X.shape[1] == len(self.selected_index)
        #         X_ = X
        #
        # elif isinstance(X, pd.DataFrame):
        #     X.columns = [i.lower() for i in X.columns]
        #     X_ = X[self.selected_fields].values
        #
        # elif isinstance(X, list):
        #     if len(X) == self.input_length:
        #         X_ = [X[i] for i, v in enumerate(X) if i in self.selected_index]
        #         X_ = np.array(X_)
        #     else:
        #         #assert len(X) == len(self.selected_index)
        #         X_ = np.array(X)
        # else:
        #     try:
        #         X = check_array(X, ensure_2d=False, force_all_finite=False)
        #         if X.shape[1] == self.input_length:
        #             X_ = X[:, self.selected_index]
        #         else:
        #             #assert X.shape[1] == len(self.selected_index)
        #             X_ = X
        #     except:
        try: # internally handled by scikit learn
            X_ = self.scaler.transform(X)
            X_ = self.lookalike_scorer.decision_function(X_)
            return X_
        except Exception as e:

            raise ValueError('{} as input is neither an numpy array, pandas data frame or a list! {}'.format(X,e))

        X_ = self.scaler.transform(X_)
        X_ = self.lookalike_scorer.decision_function(X_)
        return X_


def hist_ascii(arr, bins = 50, max_length=100, print_space=15):
    hist, bin_edges = np.histogram(arr, bins=bins)
    hist = hist / hist.max()
    hist = hist * max_length
    print('{}_|'.format(' ' * print_space))
    print('{} |'.format('{:3.4f}'.format(bin_edges[0]).center(print_space)))
    for i, v in enumerate(bin_edges[0:]):
        print('{}_| {}'.format(' ' * print_space, '*' * int(hist[i - 1])))
        print('{} |'.format('{:3.4f}'.format(v).center(print_space)))


if __name__ == '__main__':
    oneclass = OneClassLookalike(path='/Users/jitins_lab/Documents/work/DataCloud', folder_name='enh_cpg')
    t = time.time()
    print('Loading data!')
    data = pd.read_csv('/Users/jitins_lab/sources/Zeta_data/enh_cpg.csv')
    print("Data loaded in {} secs!\nTraining Model Now!".format(time.time() - t))
    t = time.time()
    oneclass.fit(data)
    print('Traning done in {} secs!\nMaking prediction!'.format(time.time() - t))
    t = time.time()
    predict = oneclass.predict(data)
    print('Prediction took {} secs!\ nSample predictions:'.format(time.time() - t))
    print(predict[:20, ])
    print('Prediction stats: \n Mean: {} Max: {} Median: {} Min: {} Std: {}'.format(
        np.mean(predict), np.max(predict), np.median(predict), np.min(predict), np.std(predict)
    ))
    print('Freq counts in the data:\n')
    
    #y, bin = np.histogram(predict, bins=50)
    # ii = np.nonzero(y)[0]
    # pred = pd.Series(predict)
    # import matplotlib.pyplot as plt
    # pred.plot.hist(grid=True, bins=50, rwidth=0.9,color='#607c8e')
    # plt.title('Bins of values')
    # plt.xlabel('Score from classifier')
    # plt.ylabel('Counts')
    # plt.grid(axis='y', alpha=0.75)
    # plt.show()
    hist_ascii(predict)
