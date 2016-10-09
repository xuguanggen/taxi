#/usr/bin/env python

from time import time
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor

import sys


from utils import load_all_data
from utils import save_results

sys.path.append('..')




Model_Name = 'rf_20161009_1'
result_csv_path = 'result/'+Model_Name+'.csv'


def run():
    ###### load data ###############
    print('1. load all data.........')
    train_fold_x,train_fold_y,val_fold_x,val_fold_y ,test_x = load_all_data()

    rf = RandomForestRegressor(
            n_estimators = 2500,
            min_samples_split = 3,
            max_depth = 16,
            n_jobs = -1
            )
     




if __name__=='__main__':
    start = time()
    run()
    end = time()
    print('Time:\t'+str((end-start)/3600)+' Hours')
