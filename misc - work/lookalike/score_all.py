from .model_script import OneClassLookalike, hist_ascii
import time
import numpy as np
import pandas as pd



def execute_all():
    # folder = str(sys.argv[1])
    folder = 'enh_cpg'
    print('Folder to look for is: {}'.format(folder))
    oneclass = OneClassLookalike(path='/home/ubuntu/ds_vol/analysis', folder_name=folder)
    t = time.time()
    print('Loading data!')
    data = pd.read_csv('/home/ubuntu/ds_vol/positive-data/{}/{}.csv'.format(folder, folder))
    print("Data loaded in {} secs!\nTraining Model Now!".format(time.time() - t))
    t = time.time()
    oneclass.fit(data)
    print('Traning done in {} secs!\nMaking prediction!'.format(time.time() - t))
    t = time.time()
    predict = oneclass.predict(data)
    print('Prediction took {} secs!\nSample predictions:'.format(time.time() - t))
    print(predict[:20, ])
    print('Prediction stats: \n Mean: {} Max: {} Median: {} Min: {} Std: {}'.format(
        np.mean(predict), np.max(predict), np.median(predict), np.min(predict), np.std(predict)
    ))
    print('Freq counts in the data:\n')
    y, bin = np.histogram(predict, bins=30)
    hist_ascii(y, bin, print_space=20)

    # adding score to profile enh_cpg
    total_pred = data.loc[:, 'email_address_md5']
    total_pred['predict_enh_cpg'] = predict

    # scoring enh_in_market
    folder = 'enh_in_market'
    print('Folder to look for is: {}'.format(folder))
    oneclass = OneClassLookalike(path='/home/ubuntu/ds_vol/analysis', folder_name=folder)
    t = time.time()
    print('Loading data!')
    data = pd.read_csv('/home/ubuntu/ds_vol/positive-data/{}/{}.csv'.format(folder, folder))
    print("Data loaded in {} secs!\nTraining Model Now!".format(time.time() - t))
    t = time.time()
    oneclass.fit(data)
    print('Traning done in {} secs!\nMaking prediction!'.format(time.time() - t))
    t = time.time()
    predict = oneclass.predict(data)
    print('Prediction took {} secs!\nSample predictions:'.format(time.time() - t))
    print(predict[:20, ])
    print('Prediction stats: \n Mean: {} Max: {} Median: {} Min: {} Std: {}'.format(
        np.mean(predict), np.max(predict), np.median(predict), np.min(predict), np.std(predict)
    ))
    print('Freq counts in the data:\n')
    y, bin = np.histogram(predict, bins=30)
    hist_ascii(y, bin, print_space=20)

    # adding scoring to
    stage_pred = data.loc[:,0]
    stage_pred['predict_enh_in_market'] = predict

    total_pred = pd.merge(total_pred,stage_pred,on='email_address_md5')

    # scoring enh retail_1
    folder = 'enh_retail_1'
    print('Folder to look for is: {}'.format(folder))
    oneclass = OneClassLookalike(path='/home/ubuntu/ds_vol/analysis', folder_name=folder)
    t = time.time()
    print('Loading data!')
    data = pd.read_csv('/home/ubuntu/ds_vol/positive-data/{}/{}.csv'.format(folder, folder))
    print("Data loaded in {} secs!\nTraining Model Now!".format(time.time() - t))
    t = time.time()
    oneclass.fit(data)
    print('Traning done in {} secs!\nMaking prediction!'.format(time.time() - t))
    t = time.time()
    predict = oneclass.predict(data)
    print('Prediction took {} secs!\nSample predictions:'.format(time.time() - t))
    print(predict[:20, ])
    print('Prediction stats: \n Mean: {} Max: {} Median: {} Min: {} Std: {}'.format(
        np.mean(predict), np.max(predict), np.median(predict), np.min(predict), np.std(predict)
    ))
    print('Freq counts in the data:\n')
    y, bin = np.histogram(predict, bins=30)
    hist_ascii(y, bin, print_space=20)

    # adding scoring to
    stage_pred = data.loc[:, 'email_address_md5']
    stage_pred['predict_enh_retail_2'] = predict

    total_pred = pd.merge(total_pred, stage_pred, on='email_address_md5')

    #scoring enh retail 2
    folder = 'enh_retail_2'
    print('Folder to look for is: {}'.format(folder))
    oneclass = OneClassLookalike(path='/home/ubuntu/ds_vol/analysis', folder_name=folder)
    t = time.time()
    print('Loading data!')
    data = pd.read_csv('/home/ubuntu/ds_vol/positive-data/{}/{}.csv'.format(folder, folder))
    print("Data loaded in {} secs!\nTraining Model Now!".format(time.time() - t))
    t = time.time()
    oneclass.fit(data)
    print('Traning done in {} secs!\nMaking prediction!'.format(time.time() - t))
    t = time.time()
    predict = oneclass.predict(data)
    print('Prediction took {} secs!\nSample predictions:'.format(time.time() - t))
    print(predict[:20, ])
    print('Prediction stats: \n Mean: {} Max: {} Median: {} Min: {} Std: {}'.format(
        np.mean(predict), np.max(predict), np.median(predict), np.min(predict), np.std(predict)
    ))
    print('Freq counts in the data:\n')
    y, bin = np.histogram(predict, bins=30)
    hist_ascii(y, bin, print_space=20)

    # adding scoring to
    stage_pred = data.loc[:, 'email_address_md5']
    stage_pred['predict_enh_retail_2'] = predict

    total_pred = pd.merge(total_pred, stage_pred, on='email_address_md5')

    # scoring enh retail 3
    folder = 'enh_retail_3'
    print('Folder to look for is: {}'.format(folder))
    oneclass = OneClassLookalike(path='/home/ubuntu/ds_vol/analysis', folder_name=folder)
    t = time.time()
    print('Loading data!')
    data = pd.read_csv('/home/ubuntu/ds_vol/positive-data/{}/{}.csv'.format(folder, folder))
    print("Data loaded in {} secs!\nTraining Model Now!".format(time.time() - t))
    t = time.time()
    oneclass.fit(data)
    print('Traning done in {} secs!\nMaking prediction!'.format(time.time() - t))
    t = time.time()
    predict = oneclass.predict(data)
    print('Prediction took {} secs!\nSample predictions:'.format(time.time() - t))
    print(predict[:20, ])
    print('Prediction stats: \n Mean: {} Max: {} Median: {} Min: {} Std: {}'.format(
        np.mean(predict), np.max(predict), np.median(predict), np.min(predict), np.std(predict)
    ))
    print('Freq counts in the data:\n')
    y, bin = np.histogram(predict, bins=30)
    hist_ascii(y, bin, print_space=20)

    # adding scoring to
    stage_pred = data.loc[:, 'email_address_md5']
    stage_pred['predict_enh_retail_3'] = predict

    total_pred = pd.merge(total_pred, stage_pred, on='email_address_md5')

    #saving file
    total_pred.to_csv('/home/ubuntu/ds_vol/analysis/onclass_output.csv')

if __name__ == '__main__':
    execute_all()