#! /usr/bin/env python


from time import time
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.append('..')
from preprocess.config import front_num_points,last_num_points,num_neighbors
from preprocess.config import train_csv_path,test_csv_path


Model_Name = 'TwoLayerRF_20160925_1'




fields = []
for i in range(front_num_points):
    fields.append('Front_LON_'+str(i+1))
    fields.append('Front_LAT_'+str(i+1))

for i in range(last_num_points):
    fields.append('Last_LON_'+str(i+1))
    fields.append('Last_LAT_'+str(i+1))


### add nearest num_neighbors destination ######
for i in range(num_neighbors):
    fields.append('Nearest_LON_'+str(i+1))
    fields.append('Nearest_LAT_'+str(i+1))

### add nearest num_neighbors odistance ######
for i in range(num_neighbors):
    fields.append('Nearest_Edistance'+str(i+1))



fields.append('TAXI_ID')
fields.append('DAYOFWEEK')
fields.append('HOUROFDAY')
fields.append('CALL_TYPE')
fields.append('DAY_TYPE')
fields.append('TRJ_Haversine_DISTANCE')
fields.append('TRJ_Euclidean_DISTANCE')
fields.append('IS_AWAY_CENTER')
fields.append('TRJ_TIME')
fields.append('TRJ_SPEED')



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



def 
