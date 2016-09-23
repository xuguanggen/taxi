#! /usr/bin/env python


from time import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.append('..')
from preprocess.config import front_num_points,last_num_points,num_neighbors
from preprocess.config import train_csv_path,test_csv_path




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

    #front_part_trj_list = []
    #last_part_trj_list = []
    #nearest_trj_destination_list = []
    #nearest_trj_edist_list = []
    #for i in range(len(df)):
    #    cur_front_part_trj = list(eval(df['TRJ_FRONT_POINTS'][i]))
    #    cur_last_part_trj = list(eval(df['TRJ_LAST_POINTS'][i]))
    #    cur_nearest_trj_destination = list(eval(df['']))
    #    cur_lonlat_front_list = []
    #    cur_lonlat_last_list = []
    #    for j in range(len(cur_front_part_trj)):
    #        cur_lonlat_front_list.append(float(cur_front_part_trj[j][0]))
    #        cur_lonlat_front_list.append(float(cur_front_part_trj[j][1]))

    #    for j in range(len(cur_last_part_trj)):
    #        cur_lonlat_last_list.append(float(cur_last_part_trj[j][0]))
    #        cur_lonlat_last_list.append(float(cur_last_part_trj[j][1]))
    #    front_part_trj_list.append(cur_lonlat_front_list)
    #    last_part_trj_list.append(cur_lonlat_last_list)
    
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




def run(result_csv_path):
    train_x,train_y = load_data(train_csv_path,True)
    test_x = load_data(test_csv_path,False)
    print('load data successfully ......')

    rf = RandomForestRegressor(
            n_estimators = 2000, #[1500,]
            min_samples_split = 2,
            max_depth = 15, # [10,]
            n_jobs = -1
            )
    rf.fit(train_x,train_y)
    y_pred = rf.predict(test_x)


    ####### save_results ###########################
    df = pd.read_csv(test_csv_path,header = 0)
    result = pd.DataFrame()
    result['TRIP_ID'] = df['TRIP_ID']
    result['LATITUDE'] = y_pred[:,1]
    result['LONGITUDE'] = y_pred[:,0]
    result.to_csv(result_csv_path,index = False)


    ###### generate report #######################
    feature_importances = rf.feature_importances_
    dic_feature_importances = dict(zip(fields,feature_importances))
    dic = sorted(dic_feature_importances.iteritems(),key = lambda d:d[1],reverse = True)
    print('feature_importances:')
    for i in range(len(dic)):
        print(dic[i][0]+":\t"+str(dic[i][1]))

if __name__=='__main__':
    start = time()
    result_csv_path = 'result/rf_20160922_1.2.csv'
    run(result_csv_path)
    end = time()
    print('Time :\t'+str((end - start)/3600) +' Hours')
