#! /usr/bin/env python

front_num_points = 5
last_num_points = 15
num_neighbors = 10



train_csv_path ="/volume1/xuguanggen/competition/taxi/data/clean_data/clean_train.csv"
test_csv_path ="/volume1/xuguanggen/competition/taxi/data/clean_data/clean_test.csv"

dict_calltype = {'A':0,'B':1,'C':2}
dict_daytype = {'A':0,'B':1,'C':2}




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
