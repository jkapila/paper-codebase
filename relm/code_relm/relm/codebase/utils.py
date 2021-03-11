import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,f1_score
from sklearn.metrics import roc_curve,cohen_kappa_score,log_loss,adjusted_mutual_info_score
from sklearn.metrics.regression import r2_score,mean_squared_error,mean_absolute_error,explained_variance_score
from scipy.stats import entropy


import matplotlib.pyplot as plt
import seaborn as sns


# objective dictionary
def mape_1(y_true, y_pred):
    abs_true = np.absolute(y_true)
    abs_pred = np.absolute(y_true - y_pred)
    n = y_true.shape[0]

    return 1 - np.sum((abs_pred / abs_true)) / n


# todo:
# 1) percentage concordant discordant
# 2) kendal's tau
# 3) gamma
# 4) k
objectives = {
    'f1_score': f1_score,
    'accuracy': accuracy_score,
    'loss': log_loss,
    'cohen_kappa': cohen_kappa_score,
    'f1_score_multi': f1_score,
    'accuracy_multi': accuracy_score,
    'loss_multi': log_loss,
    'cohen_kappa_multi': cohen_kappa_score,
    '1_mape': mape_1,
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'mi': adjusted_mutual_info_score,
    'kld': entropy
}


# funciton for plotting roc curves of models on train and test data
def plot_ROC(y_train_true, y_train_prob, y_test_true, y_test_prob, threshold=None, path ='', name=None):
    '''
    a funciton to plot the ROC curve for train labels and test labels.
    Use the best threshold found in train set to classify items in test set.
    '''
    sns.set('talk', 'whitegrid', 'dark', font_scale=1,
            rc={"lines.linewidth": 2, 'grid.linestyle': '--'})

    fpr_train, tpr_train, thresholds_train = roc_curve(y_train_true, y_train_prob, pos_label=True)
    sum_sensitivity_specificity_train = tpr_train + (1 - fpr_train)
    best_threshold_id_train = np.argmax(sum_sensitivity_specificity_train)
    best_threshold = thresholds_train[best_threshold_id_train]
    best_fpr_train = fpr_train[best_threshold_id_train]
    best_tpr_train = tpr_train[best_threshold_id_train]
    if threshold is None:
        y_train = y_train_prob >= best_threshold
    else:
        y_train = y_train_prob >= threshold

    cm_train = confusion_matrix(y_train_true, y_train)
    acc_train = accuracy_score(y_train_true, y_train)
    auc_train = roc_auc_score(y_train_true, y_train)
    f1_score_train = f1_score(y_train_true, y_train)

    print('Train Accuracy: {}'.format(acc_train))
    print('Train AUC: {}'.format(auc_train))
    print('Train F1 Score: {}'.format(f1_score_train))
    print('Train Confusion Matrix:')
    print(cm_train)

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(121)
    curve1 = ax.plot(fpr_train, tpr_train)
    curve2 = ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    dot = ax.plot(best_fpr_train, best_tpr_train, marker='o', color='black')
    ax.text(best_fpr_train, best_tpr_train, s='(%.3f,%.3f)' % (best_fpr_train, best_tpr_train))
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve (Train), AUC = %.4f' % auc_train)

    fpr_test, tpr_test, thresholds_test = roc_curve(y_test_true, y_test_prob, pos_label=True)

    if threshold is None:
        y_test = y_test_prob >= best_threshold
    else:
        y_test = y_test_prob >= threshold

    cm_test = confusion_matrix(y_test_true, y_test)
    acc_test = accuracy_score(y_test_true, y_test)
    auc_test = roc_auc_score(y_test_true, y_test)
    f1_score_test = f1_score(y_test_true, y_test)

    print('Test Accuracy: {}'.format(acc_test))
    print('Test AUC: {}'.format(auc_test))
    print('Test F1 Score: {}'.format(f1_score_test))
    print('Test Confusion Matrix:')
    print(cm_test)

    tpr_score = float(cm_test[1][1]) / (cm_test[1][1] + cm_test[1][0])
    fpr_score = float(cm_test[0][1]) / (cm_test[0][0] + cm_test[0][1])

    ax2 = fig.add_subplot(122)
    curve1 = ax2.plot(fpr_test, tpr_test)
    curve2 = ax2.plot([0, 1], [0, 1], color='navy', linestyle='--')
    dot = ax2.plot(fpr_score, tpr_score, marker='o', color='black')
    ax2.text(fpr_score, tpr_score, s='(%.3f,%.3f)' % (fpr_score, tpr_score))
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve (Test), AUC = %.4f' % auc_test)
    if len(path) != 0 and name is not None:
        place = '{}/{}.png'.format(path, 'ROC RELM Agent' if name is None else 'ROC {} Agent'.format(name))
        plt.savefig(place, dpi=500)
    plt.show()

    return best_threshold

# function to plot pair grid for latent features
def plot_pair(tr_mat):
    df_plot = pd.DataFrame(tr_mat)
    sns.set(style="ticks")
    g = sns.PairGrid(df_plot)
    g = g.map_upper(plt.scatter)
    g = g.map_lower(sns.kdeplot, cmap="Blues_d")
    g = g.map_diag(sns.kdeplot, lw=3, legend=False)
    g


def plot_learning(population_training, path='', name=None, style='seaborn-white'):
    plt.style.use(style)
    plt_data = np.array(population_training)
    iteration = plt_data.shape[0] + 1
    trainarray = np.arange(1, iteration, 1)
    ticks = np.arange(1, iteration, 10)
    if np.all(ticks != (iteration - 1)):
        ticks = np.append(ticks, iteration - 1)
    scores = -plt_data[:, 2, :]
    fig = plt.figure(figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    line_mean, = plt.plot(trainarray, np.mean(scores, axis=1))
    line_min, = plt.plot(trainarray, np.min(scores, axis=1))
    line_max, = plt.plot(trainarray, np.max(scores, axis=1))
    plt.legend([line_mean, line_min, line_max], ['mean', 'min', 'max'])
    plt.xlabel('Generations ', fontsize=20)
    plt.ylabel('Loss Socres', fontsize=16)
    plt.xticks(ticks, fontsize=14,rotation=90)
    plt.title('Log Loss across Generations', fontsize=24)
    if len(path) != 0 and name is not None:
        plt.savefig('{}/Log_loss_{}.png'.format(path, name))
    plt.show()

    ticks = np.arange(1, iteration, int(iteration / 10))
    if np.all(ticks != (iteration - 1)):
        ticks = np.append(ticks, iteration - 1)

    fig = plt.figure(figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(1, 2, 1)
    scores = plt_data[:, 0, :]
    #     line_mean, = plt.plot(trainarray, np.percentile(scores,q=25,axis=1) )
    line_min, = plt.plot(trainarray, np.median(scores, axis=1))
    line_max, = plt.plot(trainarray, np.max(scores, axis=1))
    plt.legend([line_min, line_max], ['median', 'max'])
    plt.xlabel('Generations ', fontsize=20)
    plt.ylabel('F1 Socres', fontsize=16)
    plt.xticks(ticks, fontsize=14, rotation=90)
    plt.title('F1 across Generations', fontsize=24)

    plt.subplot(1, 2, 2)
    scores = plt_data[:, 1, :]
    #     line_mean, = plt.plot(trainarray, np.percentile(scores,q=25,axis=1) )
    line_min, = plt.plot(trainarray, np.median(scores, axis=1))
    line_max, = plt.plot(trainarray, np.max(scores, axis=1))
    plt.legend([line_min, line_max], ['median', 'max'])
    plt.xlabel('Generations ', fontsize=20)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xticks(ticks, fontsize=14, rotation=90)
    plt.title('Accuracy across Generations', fontsize=24)
    plt.suptitle('Classification Behaviour of Population', fontsize=36)
    if len(path) != 0 and name is not None:
        plt.savefig('{}/Classification Behaviour_{}.png'.format(path, name))
    plt.show()

