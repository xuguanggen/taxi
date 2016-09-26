#! /usr/bin/env python

import pandas as pd
import numpy as np
import h5py
import theano


from config import Train_CSV_Path,Test_CSV_Path
from config import front_num_points,last_num_points,num_neighbors,emb_size,dis_size,MAX_LENGTH
from config import list_fields,con_fields,dis_fields

from sklearn.cluster import MeanShift,estimate_bandwidth
from keras import backend as K
from keras.preprocessing import sequence
from theano import tensor as T



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



def getConFeature(df,fieldsname):
    sub_feature = []
    for i in range(len(df)):
        sub_feature.append(float(df[fieldsname][i]))
    return np.array(sub_feature).reshape(len(df),1)




def getDisFeature(df_train,df_test,fieldsname):
    unique_ids = set()
    for i in range(len(df_train)):
        unique_ids.add(df_train[fieldsname][i])
    for i in range(len(df_test)):
        unique_ids.add(df_test[fieldsname][i])
    return dict(zip(unique_ids,range(len(unique_ids))))


def load_con_input():
    df_train = pd.read_csv(Train_CSV_Path,header = 0)
    df_test = pd.read_csv(Test_CSV_Path,header = 0)

    ###### train ###### test ###################
    tr_feature = getListFeature(df_train,list_fields[0])
    te_feature = getListFeature(df_test, list_fields[0])
    for i in range(1,len(list_fields)):
        cur_tr_feature = getListFeature(df_train,list_fields[i])
        cur_te_feature = getListFeature(df_test,list_fields[i])
        tr_feature = np.hstack([tr_feature,cur_tr_feature])
        te_feature = np.hstack([te_feature,cur_te_feature])

    for i in range(len(con_fields)):
        cur_tr_feature = getConFeature(df_train,con_fields[i])
        cur_te_feature = getConFeature(df_test,con_fields[i])
        tr_feature = np.hstack([tr_feature,cur_tr_feature])
        te_feature = np.hstack([te_feature,cur_te_feature])

    tr_label = []
    for i in range(len(df_train)):
        cur_des = list(eval(df_train['DESTINATION'][i]))
        tr_label.append(cur_des)

    tr_label = np.array(tr_label)


    tr_input = {'con_input':tr_feature,'output':tr_label}
    te_input = {'con_input':te_feature}
    return tr_input,te_input



def load_emb_input():
    df_train = pd.read_csv(Train_CSV_Path,header = 0)
    df_test = pd.read_csv(Test_CSV_Path,header = 0)
    
    tr_dis = []
    te_dis = []
    vocabs_size = []
    for i in range(len(dis_fields)):
        vocab_dict = getDisFeature(df_train,df_test,dis_fields[i])
        vocabs_size.append(len(vocab_dict))
        cur_tr_feature_ids = []
        cur_te_feature_ids = []
        for j in range(len(df_train)):
            cur_tr_feature_ids.append(vocab_dict[df_train[dis_fields[i]][j]])
        cur_tr_feature_ids = np.array(cur_tr_feature_ids,dtype=np.int32).reshape(len(df_train),dis_size)
        tr_dis.append(cur_tr_feature_ids)

        for j in range(len(df_test)):
            cur_te_feature_ids.append(vocab_dict[df_test[dis_fields[i]][j]])
        cur_te_feature_ids = np.array(cur_te_feature_ids,dtype=np.int32).reshape(len(df_test),dis_size)
        te_dis.append(cur_te_feature_ids)

    
    tr_emb_input = {'emb_input':np.hstack(tr_dis)}
    te_emb_input = {'emb_input':np.hstack(te_dis)}

    return tr_emb_input,te_emb_input,vocabs_size

def load_seq_input():
    df_train = pd.read_csv(Train_CSV_Path,header=0)
    df_test = pd.read_csv(Test_CSV_Path,header=0)
    tr_seq = []
    te_seq = []
    for i in range(len(df_train)):
        cur_trj = list(eval(df_train['POLYLINE'][i]))
        if len(cur_trj) == 1:
            tr_seq.append(cur_trj)
        else:
            tr_seq.append(cur_trj[:-1])

    for i in range(len(df_test)):
        cur_trj = list(eval(df_test['POLYLINE'][i]))
        if len(cur_trj) == 1:
            te_seq.append(cur_trj)
        else:
            te_seq.append(cur_trj[:-1])
    
    tr_seq = sequence.pad_sequences(tr_seq,maxlen=MAX_LENGTH,dtype='float32')
    te_seq = sequence.pad_sequences(te_seq,maxlen=MAX_LENGTH,dtype='float32')
    tr_seq = np.array(tr_seq)
    te_seq = np.array(te_seq)
    tr_seq_input = {'seq_input':tr_seq}
    te_seq_input = {'seq_input':te_seq}
    return tr_seq_input ,te_seq_input


#def cluster():
#    df_train = pd.read_cs(Train_CSV_Path,header=0)
#    destination = []
#    for i in range(len(df_train)):
#        destination.append(list(eval(df_train['DESTINATION'][i])))
#
#    destination = np.array(destination)
#    bw = estimate_bandwidth(
#            destination,
#            quantile = 0.1,
#            n_samples = 1000
#            )
#    ms = MeanShift(
#            bandwidth = bw,
#            bin_seeding = True,
#            min_bin_freq = 5
#            )
#    ms.fit(destination)
#    cluster_centers = ms.cluster_centers
#    with h5py.File('cluster.h5','w') as f:
#        f.create_dataset('cluster',data = cluster_centers)


def const(v):
    if theano.config.floatX=='float32':
        return np.float32(v)
    else:
        return np.float64(v)

rearth = const(6371)
deg2rad = const(3.141592653589793 / 180)


def caluate_point(inputs):
    f = h5py.File('cluster.h5')
    cluster_centers = f['cluster'][:]
    return K.dot(inputs,cluster_centers)


def hdist(y_pred,y_true):
    pred_lon = y_pred[:,0] * deg2rad
    pred_lat = y_pred[:,1] * deg2rad
    true_lon = y_true[:,0] * deg2rad
    true_lat = y_true[:,1] * deg2rad
    dlon = abs(pred_lon - true_lon)
    dlat = abs(pred_lat - true_lat)
    a1 = T.sin(dlat/2)**2 + T.cos(true_lat) * T.cos(true_lat) * (T.sin(dlon/2)**2)
    d = T.arctan2(T.sqrt(a1),T.sqrt(const(1)-a1))
    #d = T.arcsin(T.sqrt(a1))
    hd = const(2000) * rearth * d
    #return T.switch(T.eq(hd,float('nan')),(y_pred - y_true).norm(2,axis=1),hd)
    return hd



def save_results(te_predict,result_csv_path):
    df_test = pd.read_csv(Test_CSV_Path,header=0)
    result = pd.DataFrame()
    result['TRIP_ID'] = df_test['TRIP_ID']
    result['LATITUDE'] = te_predict[:,1]
    result['LONGITUDE'] = te_predict[:,0]
    result.to_csv(result_csv_path,index=False)
