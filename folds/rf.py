#/usr/bin/env python

from time import time
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor

import sys

from generate_tr_val import N_FOLDS
from utils import load_all_data
from utils import save_results
from utils import gen_report



def run():
    ###### load data ###############
    print('1. load all data.........')
    train_fold_x,train_fold_y,val_fold_x,val_fold_y,test_x = load_all_data()

    print('2. build rf model..........')
    rf = RandomForestRegressor(
            n_estimators = 2500,
            min_samples_split = 3,
            max_depth = 16,
            n_jobs = -1
            )
    
    print('3. start 10fold cv ..........')
    total_fold_error = 0
    for fold_idx in range(N_FOLDS):
        print(str(fold_idx)+' cv running..........')
        sub_tr_fold_x = train_fold_x[fold_idx]
        sub_tr_fold_y = train_fold_y[fold_idx]
        sub_val_fold_x = val_fold_x[fold_idx]
        sub_val_fold_y = val_fold_y[fold_idx]
        
        rf.fit(sub_tr_fold_x,sub_tr_fold_y)

        Model_Name = 'rf_'+str(fold_idx)
        ###### save model ########
        joblib.dump(rf,'weights/'+Model_Name+'.m')
        sub_pred_val = rf.predict(sub_val_fold_x)
        total_fold_error += gen_report(sub_val_fold_y,sub_pred_val,'log/report.log',fold_idx)
       
        pred_te = rf.predict(test_x)
        result_csv_path = 'result/rf_'+str(fold_idx)+'.csv'
        save_results(result_csv_path,pred_te)
    mean_fold_error = total_fold_error / (N_FOLDS * 1.0)
    f_report = open('log/report.log','a')
    f_report.write('Mean Fold Error:\t'+str(mean_fold_error))
    f_report.close()

if __name__=='__main__':
    start = time()
    run()
    end = time()
    print('Time:\t'+str((end-start)/3600)+' Hours')
