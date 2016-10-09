#! /usr/bin/env python

import pandas as pd
import numpy as np

from sklearn.cross_validation import KFold



import sys
sys.path.append('..')
from preprocess.config import front_num_points,last_num_points
from preprocess.config import train_csv_path,test_csv_path
from preprocess.config import fields


def getListFeature(df,fieldsname):
    sub_feature = []
    for i in range(len(df)):
        cur_list = list(eval(df[fieldsname][i]))
        cur_list = np.array(cur_list)
        cur_list = cur_list.flatten()
        
        cur_feature = []
        for j in range(len(cur_list)):
            cur_feature.append(float(cur_list[j]))
        sub_feature.append(cur_feature)

    return np.array(sub_feature)



def load_data(csv_path,isTrain):
    df = pd.read_csv(csv_path,header = 0)
    
    feature_frontTrj = getListFeature(df,'TRJ_FRONT_POINTS')
    feature_last_Trj = getListFeature(df,'TRJ_LAST_POINTS')
    feature_nearest_destination = getListFeature(df,'NEIGHBORS_DESTINATION')
    feature_nearest_odistance = getListFeature(df,'NEIGHBORS_Euclidean_DISTANCE')

    data_x = np.hstack([feature_frontTrj,feature_last_Trj,feature_nearest_destination,feature_nearest_odistance])
    for i in range( (front_num_points + last_num_points )*2 + num_neighbors*3,len(fields)):
        cur_feature_array = np.array(df[fields[i]]).reshape(data_x.shape[0],1)
        data_x = np.hstack([data_x,cur_feature_array])


    if isTrain:
        train_y = []
        for i in range(len(df)):
            train_y.append(list(eval(df['DESTINATION'][i])))
        train_y = np.array(train_y)
        return data_x,train_y
    else:
        return data_x


def load_all_data():
    train_x,train_y = load_data(train_csv_path,True)
    test_x = load_data(test_csv_path,False)
    
    ######train_x,train_y --> train_x,train_y,val_x,val_y ######
    kfold = KFold(n = train_x.shape[0],n_folds = 10,shuffle=True)

    train_fold_x = []
    train_fold_y = []
    val_fold_x = []
    val_fold_y = []
    for tr_idxs,val_idxs in kfold:
        sub_tr_fold_x = train_x[tr_idxs]
        sub_tr_fold_y = train_y[tr_idxs]
        sub_val_fold_x = train_x[val_idxs]
        sub_val_fold_y = train_y[val_idxs]

        train_fold_x.append(sub_tr_fold_x)
        train_fold_y.append(sub_tr_fold_y)
        val_fold_x.append(sub_val_fold_x)
        val_fold_y.append(sub_val_fold_y)

    return train_fold_x,train_fold_y,val_fold_x,val_fold_y,test_x







def save_results(result_csv_path,y_pred):
    df = pd.read_csv(test_csv_path,header = 0)
    result = pd.DataFrame()
    result['TRIP_ID'] = df['TRIP_ID']
    result['LATITUDE'] = y_pred[:,1]
    result['LONGITUDE'] = y_pred[:,0]
    result.to_csv(result_csv_path,index = False)


