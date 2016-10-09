#! /usr/bin/env python

from time import time
import pandas as pd
import numpy as np

from sklearn.cross_validation import KFold

import sys
sys.path.append('..')
from preprocess.config import train_csv_path,test_csv_path


N_FOLDS = 10
NUM_CUTOFF = 20


def cutoff_points(df_val):
    for idx in df_val.index:
        cur_polyline = list(eval(df_val['POLYLINE'][idx]))
        cur_len = len(cur_polyline) - 1
        cur_destination = cur_polyline[-1]
        if cur_len > NUM_CUTOFF:
            random_cutoff_len = np.random.randint(cur_len)
            cur_polyline = cur_polyline[0:cur_len - random_cutoff_len]
            cur_polyline.append(cur_destination)
            df_val['POLYLINE'][idx] = str(cur_polyline)
    return df_val



def generate_tr_val():
    df = pd.read_csv(train_csv_path,header=0)

    kfold = KFold(n=len(df),n_folds=N_FOLDS,shuffle=True)

    for (fold_idx,(tr_idxs,val_idxs)) in zip(range(N_FOLDS),kfold):
        df_train = df.iloc[tr_idxs]
        df_train.to_csv('../data/clean_data/folds_data/train_'+str(fold_idx)+'_fold.csv',index=False)

        df_val = df.loc[val_idxs]
        df_val = cutoff_points(df_val)
        df_val.to_csv('../data/clean_data/folds_data/val_'+str(fold_idx)+'_fold.csv',index=False)


def run():
    generate_tr_val()


if __name__=='__main__':
    start = time()
    run()
    end = time()
    print('Time:\t'+str((end-start)/3600)+' Hours')


