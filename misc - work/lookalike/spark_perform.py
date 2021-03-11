# dataset imports
from sklearn.datasets import load_boston, load_breast_cancer, load_iris, load_digits
from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import train_test_split

# basic dependencies
import pandas as pd
import numpy as np

# model dependencies
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

# calling spark requirements
from pyspark import SparkConf, SparkContext


# fetching data
def get_data(dataname):
    if dataname == 'iris':
        data = load_iris()
    elif dataname == 'boston':
        data = load_boston()
    elif dataname == 'cancer':
        data = load_breast_cancer()
    df = pd.concat([pd.DataFrame(data.data), pd.DataFrame(data.target)], axis=1)
    names = [i for i in data.feature_names]
    names.append('target')
    df.columns = names
    print('Data:\n')
    print(df.head())
    print('\nSummary Stats:\n')
    print(df.describe())
    return df


# getting data
df = get_data('iris')

# generating spark context
# getting spark ready
conf = (SparkConf()
        .setMaster("local[4]")
        .setAppName("class app")
        .set("spark.executor.memory", "1g"))
sc = SparkContext(conf=conf)

# splitting dta for testing
X = df.iloc[:120, 0:3].values
X_test = df.iloc[120:, 0:3].values

# putting part of data to Nearest Neighbours
nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X)

# generating oneclass classifier for the data
onecls = OneClassSVM().fit(X)

# outputs
distances, indices = nbrs.kneighbors(X_test)
print('Neighbours:\n{}\n{}'.format(distances, indices))
print('Oneclass:\n{}'.format(onecls.predict(X_test)))

# broadcasting neighbours model to spark partitions
bc_knnobj = sc.broadcast(nbrs)

# broadcasting oneclass model to spark partitions
bc_oneobj = sc.broadcast(onecls)

# getting rest data to spark rdd
# testvec = sc.parallelize([[[0,0]],[[1,1]],[[-1, -1]]])

# have to deal with this list comprehension
testvec = sc.parallelize([[i] for i in X_test.tolist()], 3)

# getting output of neighbours from the spark
results = testvec.map(lambda x: bc_knnobj.value.kneighbors(x))
print('\nNeighbour results: \n')
print(results.glom().collect())

# broadcasting oneclass model to spark partitions
bc_oneobj = sc.broadcast(onecls)

# getting output of oneclass from the spark
results2 = testvec.map(lambda x: bc_oneobj.value.predict(x))
print('\nOne Class results: \n')
print(results2.collect())


# import defined object
from model import TestPartition

# Now the test module
tst = TestPartition().fit(np.arange(10))

#broadcasting it
bc_tstobj = sc.broadcast(tst)

#testing it
print('\nTest object\'s results: \n')
print(testvec.map(lambda x: bc_tstobj.value.test_power(x,p=4)).collect())

