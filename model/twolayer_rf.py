#! /usr/bin/env python


from time import time
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor

from utils import load_data

import sys
sys.path.append('..')
from preprocess.config import front_num_points,last_num_points,num_neighbors
from preprocess.config import train_csv_path,test_csv_path
from preprocess.config import fields



Model_Name = 'TwoLayerRF_20160925_1'



def feature_engineer(feature,pred):




def run(result_csv_path):
    train_x,train_y = load_data(train_csv_path,True)
    test_x = load_data(test_csv_path,False)
    print('load data successfully.........')

    layer1_rf = RandomForestRegressor(
            n_estimators = 2500,
            max_features = 0.8,
            bootstrap = False,
            max_depth = 15,
            n_jobs = -1
            )
    layer1_rf.fit(train_x,train_y)
    print('layer 1 train successfully..........')
    ################# save model##################
    joblib.dump(layer1_rf,'weights/layer1_'+Model_Name+'.m')


