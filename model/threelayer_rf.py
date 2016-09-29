#! /usr/bin/env python


from time import time
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor

from utils import load_data
from utils import save_results


import sys
sys.path.append('..')
from preprocess.config import front_num_points,last_num_points,num_neighbors
from preprocess.config import train_csv_path,test_csv_path
from preprocess.config import fields



Model_Name = 'ThreeLayerRF_20160929_1'



def feature_engineer(rf,feature,pred):
    add_feature = []
    for sub_est in rf.estimators_:
        add_feature.append(sub_est.predict(feature))

    feature = np.hstack([feature]+add_feature+[pred])
    return feature



def run(result_csv_path):
    train_x,train_y = load_data(train_csv_path,True)
    test_x = load_data(test_csv_path,False)
    print('load data successfully.........')

    print('layer 1 train..........')
    layer1_rf = RandomForestRegressor(
            n_estimators = 2500,
            max_features = 0.8,
            bootstrap = False,
            max_depth = 15,
            n_jobs = -1
            )
    layer1_rf.fit(train_x,train_y)
    ################# save model##################
    joblib.dump(layer1_rf,'weights/layer1_'+Model_Name+'.m')

    #layer1_rf = joblib.load('weights/layer1_'+Model_Name+'.m')
    tr_pred = layer1_rf.predict(train_x)
    train_x = feature_engineer(layer1_rf,train_x,tr_pred)

    te_pred = layer1_rf.predict(test_x)
    test_x = feature_engineer(layer1_rf,test_x,te_pred)

    print('layer 2 train ............')
    layer2_rf = RandomForestRegressor(
            n_jobs = -1,
            n_estimators = 600,
            max_features = 'sqrt',
            max_depth = 20,
            bootstrap = False
            )
    layer2_rf.fit(train_x,train_y)
    joblib.dump(layer2_rf,'weights/layer2_'+Model_Name+'.m')
    
    tr_pred = layer2_rf.predict(train_x)
    train_x = feature_engineer(layer2_rf,train_x,tr_pred)
    te_pred = layer2_rf.predict(test_x)
    test_x = feature_engineer(layer2_rf,test_x,te_pred)

    print('layer 3 train ..............')
    layer3_rf = RandomForestRegressor(
            n_jobs = -1,
            n_estimators = 500,
            max_features = 'sqrt',
            max_depth = 20,
            bootstrap = False
            )
    layer3_rf.fit(train_x,train_y)
    joblib.dump(layer3_rf,'weights/layer3_'+Model_Name+'.m')
    y_pred = layer3_rf.predict(test_x)
    ############ save_results ########################
    save_results(result_csv_path,y_pred)


if __name__=='__main__':
    start = time()
    result_csv_path = 'result/'+Model_Name+'.csv'
    run(result_csv_path)
    end = time()
    print('Time :\t'+str((end - start)/3600) +' Hours')

