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


Model_Name = 'rf_20160925_1.3'




def run(result_csv_path):
    train_x,train_y = load_data(train_csv_path,True)
    test_x = load_data(test_csv_path,False)
    print('load data successfully ......')

    rf = RandomForestRegressor(
            n_estimators = 2500, #[1500,2000]
            min_samples_split = 2,
            max_depth = 16, # [10,15]
            n_jobs = -1
            )
    rf.fit(train_x,train_y)
    ###### save model ##################
    joblib.dump(rf,'weights/'+Model_Name+'.m')

    y_pred = rf.predict(test_x)


    ####### save_results ###########################
    save_results(result_csv_path,y_pred)

    ###### generate report #######################
    feature_importances = rf.feature_importances_
    dic_feature_importances = dict(zip(fields,feature_importances))
    dic = sorted(dic_feature_importances.iteritems(),key = lambda d:d[1],reverse = True)
    print('feature_importances:')
    for i in range(len(dic)):
        print(dic[i][0]+":\t"+str(dic[i][1]))

if __name__=='__main__':
    start = time()
    result_csv_path = 'result/'+Model_Name+'.csv'
    run(result_csv_path)
    end = time()
    print('Time :\t'+str((end - start)/3600) +' Hours')
