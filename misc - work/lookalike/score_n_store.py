from __future__ import print_function, division
from model_script import OneClassLookalike, hist_ascii
import time
import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib

from pyspark import SparkContext, StorageLevel
from pyspark.conf import SparkConf
from pyspark.sql import SQLContext, SparkSession
# from pyspark.sql import functions as F
from pyspark.sql.types import *

# import re, time, json, itertools, math

path = "/home/ubuntu/ds_vol/shared/"


def score_folder(path, folder):
    print('Folder to look for is: {}'.format(folder))
    oneclass = OneClassLookalike(path=os.path.join(path, 'analysis'), folder_name=folder)
    t = time.time()
    print('Loading data!')
    data = pd.read_csv(os.path.join(path, 'positive-data/{}/{}.csv'.format(folder, folder)))
    print("Data loaded in {:5.4f} secs!\nTraining Model Now!".format(time.time() - t))
    t = time.time()
    oneclass.fit(data)
    print('Traning done in {:5.4f} secs!\nMaking prediction!'.format(time.time() - t))
     
    model_path = os.path.join(path,'analysis','{}_model.dump'.format(folder))
    print('Saving model at: {}'.fromat(model_path))
    joblib.dump(oneclass,model_path)
    oneclass = joblib.load(model_path)
    t = time.time()
    predict = oneclass.predict(data)
    print('Prediction took {:5.4f} secs!\nSample predictions:'.format(time.time() - t))
    print(predict[:20, ])
    print('Prediction stats: \n Mean: {} Max: {} Median: {} Min: {} Std: {}'.format(
        np.mean(predict), np.max(predict), np.median(predict), np.min(predict), np.std(predict)
    ))
    t = time.time()
    print('Freq counts in the data:\n')
    y, bin = np.histogram(predict, bins=30)
    hist_ascii(y, bin, print_space=20)

    # adding score to profile enh_cpg
    total_pred = pd.DataFrame({'email_address_md5': data.email_address_md5,
                               'predict_enh_cpg': predict})

    op_path = os.path.join(path, os.path.join('analysis', '{}_positive.csv'.format(folder)))

    total_pred.to_csv(op_path)
    print('\nOutput saved in {:5.4f} secs at:\n{}'.format(time.time() - t, op_path))
    return oneclass


def score_lookalike(sc, path, folder, scorer):
    t = time.time()
    # rdd_df = sc.textFile(os.path.join(path, 'shared', folder, (folder + '.csv*.gz'))) \
    # rdd_df = sc.textFile(os.path.join(path, 'shared', folder, (folder + '.csv*.gz'))) \
    rdd_df = sc.textFile(os.path.join(path, 'shared', folder, (folder + '.50k.gz'))) \
        .map(lambda l: l.replace('"', '').strip().split(",")) \
        .filter(lambda x: len(x) == len(header))

    header = pd.read_csv(os.path.join(path, 'shared', folder, (folder + '.header'))).columns
    schema = StructType([StructField(i, StringType(), True) for i in header])

    # keep_cols=dedup_cols[dedup_cols['FOLDER']==folder].FIELD_NAME
    # keep_cols = cont_cols[cont_cols['FOLDER'] == folder].FIELD_NAME_HEADER

    df = spark.createDataFrame(rdd_df, schema)

    # df = df.repartition(50).persist(StorageLevel.MEMORY_AND_DISK)
    print('Scoring data loaded in {:5.4f} secs'.format(time.time()-t))
    t = time.time()
    lookalikobj = sc.broadcast(oneclass)

    results = df.select(list(scorer.selected_fields)).rdd
    print(results.take(20))
    results = results.map(lambda x: lookalikobj.value.predict(x))
    scores = results.collect()
    print(scores)
    # print(scores.)
    md5s = df.select(['email_address_md5'])

    lookalike_scores = pd.DataFrame({'email_address_md5': md5s,
                                     'lookalike_score': scores})
    op_path = os.path.join(path, os.path.join('analysis', '{}_alike.csv'.format(folder)))

    lookalike_scores.to_csv(op_path)
    print('\nOutput saved in {:5.4f} secs at:\n{}'.format(time.time() - t, op_path))
    return oneclass


if __name__ == '__main__':
    conf = SparkConf().setAppName("ScorerAnalysis")
    conf = (conf.setMaster('local[6]')
            .set('spark.executor.memory', '3G')
            .set('spark.driver.memory', '3G')
            .set('spark.driver.maxResultSize', '3G')
            .set('spark.debug.maxToStringFields', 1000))

    # sc = SparkContext(conf=conf)

    spark = SparkSession.builder.master("local[4]").appName("ScorerAnalysis").config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    sqlContext = SQLContext(sc)
    print('Spark standalone with properties:\n', sc.getConf().getAll())

    oneclass = score_folder('/Users/jitins_lab/sources/ins_lookalike', 'enh_cpg')

    print('Scoring for: {}'.format('enh_cpg'))
    score_lookalike(sc, '/Users/jitins_lab/sources/ins_lookalike', 'enh_cpg', oneclass )
    sc.cancelAllJobs()