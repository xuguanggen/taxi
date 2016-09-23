#! /usr/bin/env python
import os

DIR='/home/xuguanggen/competition/taxi'

Train_CSV_Path='/home/xuguanggen/competition/taxi/shanghai/data/clean_train.csv'
Test_CSV_Path='/home/xuguanggen/competition/taxi/shanghai/data/clean_test.csv'

front_num_points = 5
last_num_points = 15
num_neighbors = 10

emb_size = 10
dis_size = 1

list_fields = [
    'TRJ_FRONT_POINTS',
    'TRJ_LAST_POINTS',
    'NEIGHBORS_DESTINATION',
    'NEIGHBORS_Euclidean_DISTANCE'
    ]
con_fields = [
        'TRJ_Haversine_DISTANCE',
        'TRJ_Euclidean_DISTANCE',
        'TRJ_TIME',
        'TRJ_SPEED'
    ]
dis_fields = [
    'TAXI_ID',
    'DAYOFWEEK',
    'HOUROFDAY',
    'IS_AWAY_CENTER'
    ]




