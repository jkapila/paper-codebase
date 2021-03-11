from __future__ import print_function, division
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.externals import joblib
from bt_ai.experimental.lookalike.oneclass.model_script import OneClassLookalike, hist_ascii
from bt_ai.experimental.lookalike.oneclass.scorer import CloudPickleBatchScorer


def plot_hist(pred_train, pred_test, pred_valid, ext_perc=1.25, n_bins=20, name=''):
    actuals = pred_train
    tested = pred_test
    validated = pred_valid

    if actuals is None:
        return

    min_val = np.min([np.min(actuals), np.min(tested)])
    max_val = np.max([np.max(actuals), np.max(tested)])
    # n_bins = 20
    binswidth = np.arange(min_val * ext_perc, max_val * ext_perc, (max_val - min_val) / n_bins)
    bincenters = 0.5 * (binswidth[1:] + binswidth[:-1])
    hist_a, _ = np.histogram(actuals, bins=binswidth)
    hist_t, _ = np.histogram(tested, bins=binswidth)

    plt.style.use('seaborn-white')
    fig, ax1 = plt.subplots(figsize=(20, 15))

    color = 'tab:red'
    ax1.set_xlabel('Range of Values')
    ax1.set_xticks(bincenters)
    ax1.set_ylabel('Actuals', color=color, fontsize=14)
    ax1.plot(bincenters, hist_a, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Tested', color=color, fontsize=14)  # we already handled the x-label with ax1
    ax2.plot(bincenters, hist_t, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    if validated is not None:
        hist_v, _ = np.histogram(validated, bins=binswidth)
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        color = 'tab:green'
        ax3.set_ylabel('Validated', color=color, fontsize=14)  # we already handled the x-label with ax1
        ax3.plot(bincenters, hist_v, color=color)
        ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Histogram plot of Datasets sets for {}'.format(name), fontsize=28)
    plt.savefig('Histogram_{}.png'.format(name))
    # plt.show()


def train_and_score(sub_data, data, df_part, model='svm', name='default'):
    oneclass = OneClassLookalike(path=os.path.join(path, 'analysis'), folder_name=folder, method=model, limit=0.001)
    t = time.time()
    oneclass.fit(sub_data)
    print('Training done in {:5.4f} secs!'.format(time.time() - t))

    CloudPickleBatchScorer(oneclass).dump('oneclass_{}_{}.gz'.format(model, name))

    t = time.time()
    try:
        preds = oneclass.score(sub_data)
        preds2 = oneclass.score(data[oneclass.selected_fields].values)
        if df_part is not None:
            preds3 = oneclass.score(df_part.values[:, oneclass.selected_index])
            print('Making prediction took {:5.4f} secs'.format(time.time() - t))
            return preds, preds2, preds3
        else:
            print('Making prediction took {:5.4f} secs'.format(time.time() - t))
            return preds, preds2, None
    except:
        return None, None, None


if __name__ == '__main__':
    # setting path variables
    path = '/Users/jitins_lab/sources/ins_lookalike'
    # path = '/home/ubuntu/ds_vol'
    folder = 'enh_cpg'

    # loading data
    t = time.time()
    print('Loading data!')
    data_m = pd.read_csv(os.path.join(path, 'positive-data/{}/{}.csv'.format(folder, folder)), low_memory=False)
    data_mrk = pd.read_csv(os.path.join(path, 'positive-data/enh_in_market/enh_in_market.csv'.format(folder, folder)),
                           low_memory=False)
    data_mde = pd.read_csv(os.path.join(path, 'positive-data/enh_mde_orig/enh_mde_orig.csv'.format(folder, folder)),
                           low_memory=False)
    data_r1 = pd.read_csv(os.path.join(path, 'positive-data/enh_retail_1/enh_retail_1.csv'.format(folder, folder)),
                          low_memory=False)
    data_r2 = pd.read_csv(os.path.join(path, 'positive-data/enh_retail_2/enh_retail_2.csv'.format(folder, folder)),
                          low_memory=False)
    data_r3 = pd.read_csv(os.path.join(path, 'positive-data/enh_retail_3/enh_retail_3.csv'.format(folder, folder)),
                          low_memory=False)

    data = pd.merge(data_m, data_r1, on='email_address_md5')
    data = pd.merge(data, data_r2, on='email_address_md5')
    data = pd.merge(data, data_r3, on='email_address_md5')
    data = pd.merge(data, data_mrk, on='email_address_md5')
    data = pd.merge(data, data_mde, on='email_address_md5')
    print(data.shape)

    print("Data loaded in {:5.4f} secs!".format(time.time() - t))
    print(data.head())

    # scoring on part of 70M
    # df_part = pd.read_csv('/Users/jitins_lab/sources/ins_lookalike/shared/enh_cpg/enh_cpg.csv0000_part_00.gz',
    #                       compression='gzip', nrows=20000, header=None)

    df_part = None

    t = time.time()
    print('Loading Profiles!')
    profiles = pd.read_csv(os.path.join(path, 'feedback-data/{}'.format('LYFT')), header=None)
    print("Profiles of shape: {} loaded in {:5.4f} secs!".format(profiles.shape, time.time() - t))
    profiles.columns = ['email_address_md5', 'audience', 'identifier', 'click', 'strt_time', 'stop_time']
    print(profiles.head())

    # subseting data
    sub_prof = profiles[(profiles.audience == 'lyft_7') & (profiles.click == 'click')].email_address_md5
    sub_data = data[data.email_address_md5.isin(sub_prof)]
    print(sub_data.shape)

    scores = train_and_score(sub_data, data, df_part, model='if', name='lyft')
    plot_hist(scores[0], scores[1], scores[2], name='LYFT with Isolation Forest')

    scores = train_and_score(sub_data, data, df_part, model='svm', name='lyft')
    plot_hist(scores[0], scores[1], scores[2], name='LYFT with Oneclass SVM')

    t = time.time()
    print('Loading Profiles!')
    profiles = pd.read_csv(os.path.join(path, 'feedback-data/{}'.format('CREDIT_CARD')), header=None)
    print("Profiles loaded in {:5.4f} secs!".format(time.time() - t))
    profiles.columns = ['email_address_md5', 'audience', 'identifier', 'click', 'strt_time', 'stop_time']
    print(profiles.head())

    # subseting data
    sub_prof = profiles[(profiles.audience == 'creditcard_4') & (profiles.click == 'click')].email_address_md5
    sub_data = data[data.email_address_md5.isin(sub_prof)]
    print(sub_data.shape)

    scores = train_and_score(sub_data, data, df_part, model='if', name='ccard')
    plot_hist(scores[0], scores[1], scores[2], name='CREDIT_CARD with Isolation Forest')

    scores = train_and_score(sub_data, data, df_part, model='svm', name='ccard')
    plot_hist(scores[0], scores[1], scores[2], name='CREDIT_CARD with Oneclass SVM')

    t = time.time()
    print('Loading Profiles!')
    profiles = pd.read_csv(os.path.join(path, 'feedback-data/{}'.format('AUTO_INSURANCE')), header=None)
    print("Profiles loaded in {:5.4f} secs!".format(time.time() - t))
    profiles.columns = ['email_address_md5', 'audience', 'identifier', 'click', 'strt_time', 'stop_time']
    print(profiles.head())

    # subseting data
    sub_prof = profiles[(profiles.audience == 'autoinsurance_4') & (profiles.click == 'click')].email_address_md5
    sub_data = data[data.email_address_md5.isin(sub_prof)]
    print(sub_data.shape)

    scores = train_and_score(sub_data, data, df_part, model='if', name='auto')
    plot_hist(scores[0], scores[1], scores[2], name='AUTO_INSURANCE with Isolation Forest')

    scores = train_and_score(sub_data, data, df_part, model='svm', name='auto')
    plot_hist(scores[0], scores[1], scores[2], name='AUTO_INSURANCE with Oneclass SVM')
