#! /usr/bin/env python

import pandas as pd
import numpy as np
import cPickle as pickle
import math as Math
import time as Time
from time import time

from sklearn.neighbors import NearestNeighbors

from utils import compute_hdistance,compute_odistance
from config import front_num_points,last_num_points,num_neighbors
from config import train_csv_path,test_csv_path

min_lon = -8.8
max_lon = -7
min_lat = 41
max_lat = 42

center_lon = (min_lon + max_lon) / 2
center_lat = (min_lat + max_lat) / 2
####### lon axis= 150km,lat axis = 110km ###########
#### each grid side length is 500m ###############
grid_length = 500
lon_step = 0.006
lat_step = 0.0045454546
numsOfgrid_lon_axis = int(Math.ceil((max_lon - min_lon) / lon_step))
numsOfgrid_lat_axis = int(Math.ceil((max_lat - min_lat) / lat_step))




def map2grid():
    lon_begin = min_lon + lon_step
    lat_begin = min_lat + lat_step

    lon_axis_idx = 1
    lat_axis_idx = 1

    grid_idx = 1
    lon_grid_idx_dict = {}
    lat_grid_idx_dict = {}
    while lat_axis_idx <= numsOfgrid_lat_axis:
        while lon_axis_idx <= numsOfgrid_lon_axis:
            lon_grid_idx_dict[grid_idx] = lon_begin
            lat_grid_idx_dict[grid_idx] = lat_begin
            grid_idx += 1
            lon_begin += lon_step
            lon_axis_idx += 1
        lon_begin = min_lon + lon_step
        lat_begin += lat_step
        lon_axis_idx = 1
        lat_axis_idx += 1

    with open('map/lon_grid_idx_dict.pkl','w') as f_lon_grid_idx_dict:
        pickle.dump(lon_grid_idx_dict,f_lon_grid_idx_dict,protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('map/lat_grid_idx_dict.pkl','w') as f_lat_grid_idx_dict:
        pickle.dump(lat_grid_idx_dict,f_lat_grid_idx_dict,protocol=pickle.HIGHEST_PROTOCOL)


def trj2grid_idx(csv_path):
    df = pd.read_csv(csv_path,header = 0)
    trj_idx_list = []
    for i in range(len(df)):
        trj_data = list(eval(df["POLYLINE"][i]))
        trj_idx = []
        for j in range(len(trj_data)):
            lon = float(trj_data[j][0])
            lat = float(trj_data[j][1])
            grid_rowidx = int(Math.floor((lat - min_lat)/(lat_step)))
            grid_colidx = int(Math.ceil((lon - min_lon)/(lon_step)))
            grid_idx = int(grid_rowidx * numsOfgrid_lon_axis + grid_colidx)
            trj_idx.append(grid_idx)
        trj_idx_list.append(str(trj_idx))
    df['TRJ_IDX'] = trj_idx_list
    df.to_csv(csv_path,index=False)


def timestamp2features(csv_path):
    df = pd.read_csv(csv_path,header = 0)
    ############## the day of week####################
    weekdayList = []
    ############# the hour of a day ##################
    hourList = []
    for i in range(len(df)):
        cur_time = df["TIMESTAMP"][i]
        cur_time = Time.localtime(cur_time)
        w = int(Time.strftime('%w',cur_time))
        h = int(Time.strftime('%H',cur_time))
        weekdayList.append(w)
        hourList.append(h)
    df['DAYOFWEEK'] = weekdayList
    df['HOUROFDAY'] = hourList
    df.to_csv(csv_path,index = False)



###### select front num_points trj_grid_idx and last num_points trj_grid_idx ########
def generate_firstPartTrjData(csv_path,num_points,isTrain):
    df = pd.read_csv(csv_path)
    firstPart_data_list  = []
    destination_list = []

    for i in range(len(df)):
        cur_trj = df["POLYLINE"][i]
        cur_trj = list(eval(cur_trj))
        if isTrain:
            destination_list.append(cur_trj[-1])

        if len(cur_trj) == 1:
            firstPart_data_list.append(num_points * [cur_trj[-1]])
        else:
            if isTrain:
                cur_trj = cur_trj[:-1]
            cur_firstpart = cur_trj[0:num_points]
            if len(cur_firstpart) < num_points:
                firstpart_len = len(cur_firstpart)
                for j in range(num_points - firstpart_len):
                    cur_firstpart.append(cur_trj[-1])
            firstPart_data_list.append(cur_firstpart)
    df['TRJ_FRONT_POINTS'] = firstPart_data_list
    if isTrain:
        df['DESTINATION'] = destination_list
    df.to_csv(csv_path,index = False)

def generate_lastPartTrjData(csv_path,num_points,isTrain):
    df = pd.read_csv(csv_path,header = 0)
    lastPart_list = []
    for i in range(len(df)):
        cur_trj = list(eval(df['POLYLINE'][i]))
        if len(cur_trj) == 1:
            lastPart_list.append(num_points * [cur_trj[-1]])
        else:
            if isTrain:
                cur_trj = cur_trj[:-1]
            cur_lastpart = cur_trj[-num_points:]
            if len(cur_lastpart) < num_points:
                lastpart_len = len(cur_lastpart)
                for j in range(num_points - lastpart_len):
                    cur_lastpart.append(cur_trj[-1])
            lastPart_list.append(cur_lastpart)
    df['TRJ_LAST_POINTS'] = lastPart_list
    df.to_csv(csv_path,index = False)

def gen_trjfeature(csv_path):
    df = pd.read_csv(csv_path,header = 0)
    h_distance_list = []
    o_distance_list = []
    lastPoint_destination_distance_list = []
    trj_time_list = []
    speed_list = []
    isDriveAwayCity_list = []
    for i in range(len(df)):
        cur_trj_lonlat_list = list(eval(df["POLYLINE"][i]))
        if len(cur_trj_lonlat_list) <= 2:
            h_distance_list.append(0)
            o_distance_list.append(0)
            speed_list.append(0)
            isDriveAwayCity_list.append(0)
        else:
            first_lon = cur_trj_lonlat_list[0][0]
            first_lat = cur_trj_lonlat_list[0][1]
            last_lon = cur_trj_lonlat_list[-2][0]
            last_lat = cur_trj_lonlat_list[-2][1]
            hdistance = compute_hdistance(first_lon,first_lat,last_lon,last_lat)
            odistance = compute_odistance(first_lon,first_lat,last_lon,last_lat)
            h_distance_list.append(hdistance)
            o_distance_list.append(odistance)

            first_center_distance = compute_hdistance(first_lon,first_lat,center_lon,center_lat)
            last_center_distance = compute_hdistance(last_lon,last_lat,center_lon,center_lat)
            if first_center_distance >= last_center_distance:
                isDriveAwayCity_list.append(0)
            else:
                isDriveAwayCity_list.append(1)

            cur_speed = hdistance / (15 * (len(cur_trj_lonlat_list) - 1))
            speed_list.append(cur_speed)

        if len(cur_trj_lonlat_list) < 2:
            #lastPoint_destination_distance_list.append(0)
            trj_time_list.append(0)
        else:
            #destination_lon = cur_trj_lonlat_list[-1][0]
            #destination_lat = cur_trj_lonlat_list[-1][1]
            #lastPoint_lon = cur_trj_lonlat_list[-2][0]
            #lastPoint_lat = cur_trj_lonlat_list[-2][1]
            #distance = compute_hdistance(lastPoint_lon,lastPoint_lat,destination_lon,destination_lat)
            #lastPoint_destination_distance_list.append(distance)

            trj_time_list.append(15 * (len(cur_trj_lonlat_list)))

    df["TRJ_Haversine_DISTANCE"] = h_distance_list
    df['TRJ_Euclidean_DISTANCE'] = o_distance_list
    df['IS_AWAY_CENTER'] = isDriveAwayCity_list
    #df["LAST_DISTANCE"] = lastPoint_destination_distance_list
    df["TRJ_TIME"] = trj_time_list
    df["TRJ_SPEED"] = speed_list
    df.to_csv(csv_path,index = False)


def gen_gridfeature(csv_path):
    df = pd.read_csv(csv_path,header = 0)
    df_length = len(df)
    grid_nums = {}
    startPoint_nums = {}
    endPoint_nums = {}
    for i in range(df_length):
        cur_trj_idx_list = list(eval(df['TRJ_IDX'][i]))
        startPoint = cur_trj_idx_list[0]
        endPoint = cur_trj_idx_list[-1]
        for cur_trj_idx in cur_trj_idx_list:
            if cur_trj_idx in grid_nums.keys():
                grid_nums[cur_trj_idx] = grid_nums[cur_trj_idx] + 1
            else:
                grid_nums[cur_trj_idx] = 1

        if startPoint in startPoint_nums.keys():
            startPoint_nums[startPoint] = startPoint_nums[startPoint] + 1
        else:
            startPoint_nums[startPoint] = 1

        if endPoint in endPoint_nums.keys():
            endPoint_nums[endPoint] = endPoint_nums[endPoint] + 1
        else:
            endPoint_nums[endPoint] = 1
    
    df = pd.DataFrame()
    grid_nums_list = []
    startPoint_nums_list = []
    endPoint_nums_list = []
    for grid_idx in range(1, numsOfgrid_lat_axis*numsOfgrid_lon_axis + 1):
        if grid_idx in grid_nums.keys():
            grid_nums_list.append(grid_nums[grid_idx])
        else:
            grid_nums_list.append(0)

        if grid_idx in startPoint_nums.keys():
            startPoint_nums_list.append(startPoint_nums[grid_idx])
        else:
            startPoint_nums_list.append(0)

        if grid_idx in endPoint_nums.keys():
            endPoint_nums_list.append(endPoint_nums[grid_idx])
        else:
            endPoint_nums_list.append(0)

    df['grid_idx']= range(1,numsOfgrid_lat_axis * numsOfgrid_lon_axis +1)
    df['grid_nums'] = grid_nums_list
    df['startPoint_nums'] = startPoint_nums_list
    df['endPoint_nums'] = endPoint_nums_list

    df.to_csv('map/grid_feature.csv',index=False)



def generater_neighbours_feature():
    df_train = pd.read_csv(train_csv_path,header = 0)
    df_test = pd.read_csv(test_csv_path,header = 0)

    train_trj_lastpart = []
    train_destination = []
    test_trj_lastpart = []

    for i in range(len(df_train)):
        train_trj_lastpart.append(list(eval(df_train['TRJ_LAST_POINTS'][i])))
        train_destination.append(list(eval(df_train['DESTINATION'][i])))

    for i in range(len(df_test)):
        test_trj_lastpart.append(list(eval(df_test['TRJ_LAST_POINTS'][i])))

    train_trj_lastpart = np.array(train_trj_lastpart).reshape(len(df_train),last_num_points * 2)
    train_destination = np.array(train_destination)
    test_trj_lastpart = np.array(test_trj_lastpart).reshape(len(df_test),last_num_points * 2)

    neigh = NearestNeighbors(num_neighbors + 1,0.4)
    neigh.fit(train_trj_lastpart)
    
    train_neighbors_idx_odistance = neigh.kneighbors(train_trj_lastpart)
    test_neighbors_idx_odistance = neigh.kneighbors(test_trj_lastpart)


    train_neigh_odist_list = []
    train_neigh_destination_list = []

    for i in range(len(df_train)):
        cur_neigh_idx = train_neighbors_idx_odistance[1,i,1:]
        cur_neigh_odist = train_neighbors_idx_odistance[0,i,1:]
        cur_neigh_destination = train_destination[cur_neigh_idx].reshape(2 * num_neighbors)
        train_neigh_odist_list.append(cur_neigh_odist)
        train_neigh_destination_list.append(cur_neigh_destination)

    test_neigh_odist_list = []
    test_neigh_destination_list = []
    for i in range(len(df_test)):
        cur_neigh_idx = test_neighbors_idx_odistance[1,i,1:]
        cur_neigh_odist = test_neighbors_idx_odistance[0,i,1:]
        cur_neigh_destination = train_destination[cur_neigh_idx].reshape(2 * num_neighbors)
        test_neigh_odist_list.append(cur_neigh_odist)
        test_neigh_destination_list.append(cur_neigh_destination)

    df_train['NEIGHBORS_Euclidean_DISTANCE'] = train_neigh_odist_list
    df_train['NEIGHBORS_DESTINATION'] = train_neigh_destination_list

    df_test['NEIGHBORS_Euclidean_DISTANCE'] = test_neigh_odist_list
    df_test['NEIGHBORS_DESTINATION'] = test_neigh_destination_list

    df_train.to_csv(train_csv_path,index=False)
    df_test.to_csv(test_csv_path,index=False)

def run():
    map2grid()
    print("1----map2grid completed ....")
    
    #trj2grid_idx(train_csv_path)
    #trj2grid_idx(test_csv_path)
    #print("2---trj2grid_idx train test completed......")

    generate_firstPartTrjData(train_csv_path, front_num_points,True)
    generate_firstPartTrjData(test_csv_path, front_num_points,False)
    print("3---generate_firstPartTrjData train test completed .....")
    
    generate_lastPartTrjData(train_csv_path, last_num_points,True)
    generate_lastPartTrjData(test_csv_path, last_num_points,False)
    print("4---generate_lastPartTrjData train test completed .....")
    
    #timestamp2features(train_csv_path)
    #timestamp2features(test_csv_path)
    #print("5---timestamp2features train test completed.....")

    #gen_trjfeature(train_csv_path)
    #gen_trjfeature(test_csv_path)
    #print("6---generate trj feature train test completed......")

    #gen_gridfeature(train_csv_path)
    #print("6---generate grid feature train completed......")

if __name__=='__main__':
    start = time()
    run()
    end  = time()
    print("Time :"+str((end - start)/3600)+" Hours")
