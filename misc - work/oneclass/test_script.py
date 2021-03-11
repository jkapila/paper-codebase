from bt_ai.experimental.lookalike.oneclass.scorer import CloudPickleBatchScorer
from bt_ai.experimental.lookalike.oneclass.model_script import OneClassLookalike
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    folder = 'enh_cpg'
    path = '/Users/jitins_lab/sources/ins_lookalike'
    data_m = pd.read_csv(os.path.join(path, 'positive-data/{}/{}.csv'.format(folder, folder)), low_memory=False)
    print('Data Loaded!')
    model = OneClassLookalike(path=os.path.join(path, 'analysis'), folder_name='acxiom', limit=0.001)
    model.fit(data_m)
    user_data = [(12341, {"ap006095": 1, "ap006094": 1}),
                 (12342, {"ap006095": 10, "ap006094": 10}),
                 (12343, {"ap006093": 20, "ap006094": 20}),
                 (12344, {"ap006093": 1, "ap006095": 1}),
                 (12345, {"ap006095": 0, "ap006094": 0}),
                 (12346, {"ap006093": 2, "ap006094": 1}),
                 (12347, {"ap006093": 0, "ap006095": 2}),
                 (12348, {"ap006095": 0, "ap006094": 2}),
                 (12349, {"ap006093": 1, "ap006094": 1}),
                 ]
    print('Model Made')
    print(model.score(user_data))
    print(model.scorer_hints())

    CloudPickleBatchScorer(model).dump('oneclass_{}_{}.gz'.format(model.method, 'overall'))
    print('Model Saved')
    model = CloudPickleBatchScorer.load('/Users/jitins_lab/sources/ml-models/bt_ai/experimental/lookalike/oneclass/oneclass_if_overall.gz')

    print("Model Loaded!")
    print(model.scorer_hints())
    print(model.score(user_data))