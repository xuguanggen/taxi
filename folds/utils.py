#! /usr/bin/env python

import pandas as pd
import numpy as np
import math as Math


from sklearn.cross_validation import KFold

from generate_tr_val import N_FOLDS

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
    test_x = load_data(test_csv_path,False)

    train_fold_x = []
    train_fold_y = []
    val_fold_x = []
    val_fold_y = []
    for fold_idx in range(N_FOLDS):
        sub_tr_fold_x,sub_tr_fold_y = load_data('../data/clean_data/folds_data/train_'+str(fold_idx)+'_fold.csv',True)
        sub_val_fold_x,sub_tr_fold_y = load_data('../data/clean_data/folds_data/val_'+str(fold_idx)+'_fold.csv',True)

        train_fold_x.append(sub_tr_fold_x)
        train_fold_y.append(sub_tr_fold_y)
        val_fold_x.append(sub_val_fold_x)
        val_fold_y.append(sub_val_fold_y)

    return train_fold_x,train_fold_y,val_fold_x,val_fold_y,test_x

def rad(d):
    return d * Math.pi /180.0

###  return hdistance unit is miles ################
def compute_hdistance(lon1,lat1,lon2,lat2):
    lon1 = float(lon1)
    lat1 = float(lat1)
    lon2 = float(lon2)
    lat2 = float(lat2)

    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lon1) - rad(lon2)
    s = 2 * Math.asin(Math.sqrt(Math.pow(Math.sin(a/2),2) + 
        Math.cos(radLat1) * Math.cos(radLat2) * Math.pow(Math.sin(b/2),2)))
    s = s * 6378.137
    s = round(s * 10000) /10000
    return s

def gen_report(true_pt,pred_pt,report_path,fold_idx):
    sum_error = 0
    min_error = 10000000
    max_error = 0
    for i in range(true_pt.shape[0]):
        cur_error = compute_error(true_pt[i][0],true_pt[i][1],pred_pt[i][0],pred_pt[i][1])
        min_error = min(min_error,cur_error)
        max_error = max(max_error,cur_error)
        sum_error += cur_error

    avg_error = sum_error / true_pt.shape[0]
    f_report = open(report_path,'a')
    f_report.write('====================================')
    f_report.write('Fold:\t'+str(fold_idx)+'\n')
    f_report.write('min_error:\t'+str(min_error)+'\n')
    f_report.write('max_error:\t'+str(max_error)+'\n')
    f_report.write('mean_error:\t'+str(avg_error)+'\n\n\n')
    f_report.close()
    return avg_error


def save_results(result_csv_path,y_pred):
    df = pd.read_csv(test_csv_path,header = 0)
    result = pd.DataFrame()
    result['TRIP_ID'] = df['TRIP_ID']
    result['LATITUDE'] = y_pred[:,1]
    result['LONGITUDE'] = y_pred[:,0]
    result.to_csv(result_csv_path,index = False)


