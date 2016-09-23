#! /usr/bin/env python

from config import Train_CSV_Path,Test_CSV_Path
from config import dis_size,emb_size
from utils import load_con_input,load_emb_input



from sklearn.cluster import MeanShift,estimate_bandwidth
import h5py
import numpy as np
import pandas as pd
import math as Math
from time import time
import cPickle as pickle

from keras.models import model_from_json,Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten,Merge,Lambda
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD,Adagrad
from keras.layers import BatchNormalization



def prepare_inputX(con_feature,emb_feature):
    n_emb = emb_feature.shape[1]
    input_x = []
    input_x.append(con_feature)
    for i in range(n_emb):
        input_x.append(emb_feature[:,i])
    return input_x


def cluster():
    df_train = pd.read_csv(Train_CSV_Path,header=0)
    destination = []
    for i in range(len(df_train)):
        destination.append(list(eval(df_train['DESTINATION'][i])))

    destination = np.array(destination)
    bw = estimate_bandwidth(
            destination,
            quantile = 0.1,
            n_samples = 1000
            )
    ms = MeanShift(
            bandwidth = bw,
            bin_seeding = True,
            min_bin_freq = 5
            )
    ms.fit(destination)
    cluster_centers = ms.cluster_centers_
    with h5py.File('cluster.h5','w') as f:
        f.create_dataset('cluster',data = cluster_centers)
    return cluster_centers

def load_dataset():
    tr_input,te_input = load_con_input()
    tr_con_feature = tr_input['con_input']
    tr_label = tr_input['output']
    te_con_feature = te_input['con_input']

    tr_emb_input ,te_emb_input,vocabs_size = load_emb_input()
    tr_emb_feature = tr_emb_input['emb_input']
    te_emb_feature = te_emb_input['emb_input']

    return tr_con_feature,tr_emb_feature,tr_label,te_con_feature,te_emb_feature,vocabs_size

def build_mlp(n_con,n_emb,vocabs_size,n_dis,emb_size,cluster_size):
    hidden_size = 800
    con = Sequential()
    con.add(Dense(input_dim=n_con,output_dim=emb_size))

    emb_list = []
    for i in range(n_emb):
        emb = Sequential()
        emb.add(Embedding(input_dim=vocabs_size[i],output_dim=emb_size,input_length=n_dis))
        emb.add(Flatten())
        emb_list.append(emb)

    model = Sequential()
    model.add(Merge([con] + emb_list,mode='concat'))
    model.add(BatchNormalization())
    model.add(Dense(hidden_size,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(cluster_size,activation='softmax'))
    model.add(Lambda(caluate_point, output_shape =[2]))
    return model





def main(result_csv_path,hasCluster):
    print('1. Loading Data.........')
    tr_con_feature,tr_emb_feature,tr_label,te_con_feature,te_emb_feature,vocabs_size = load_dataset()
    
    n_con = tr_con_feature.shape[1]
    n_emb = tr_emb_feature.shape[1]
    
    train_x = prepare_inputX(tr_con_feature,tr_emb_feature)
    test_x = prepare_inputX(te_con_feature,te_emb_feature)
    print('1.1 cluster.............')
    cluster_centers = []
    if hasCluster:
        f = h5py.File('cluster.h5','r')
        cluster_centers = f['cluster'][:]
    else:
        cluster_centers = cluster()

    print('2. Building model..........')
    model_name = 'MLP_0.1'
    model = build_mlp(n_con,n_emb,vocabs_size,dis_size,emb_size,cluster_centers.shape[0])
    model.compile(loss='mean_squared_error',optimizer=Adagrad())
    model.fit(
        train_x,
        tr_label,
        nb_epoch = 200,
        batch_size = 500,
        verbose = 1,
        validation_split = 0.3,
    )
    ##### dump model ########
    json_string = model.to_json()
    open('weights/'+ model_name +'.json','w').write(json_string)
    model.save_weights('weights/'+ model_name + '.h5',overwrite=True)

    ####### predict #############################
    print('3. Predicting result.........')
    te_predict = model.predict(test_x)
    df_test = pd.read_csv(Test_CSV_Path,header=0)
    result = pd.DataFrame()
    result['TRIP_ID'] = df_test['TRIP_ID']
    result['LATITUDE'] = te_predict[:,1]
    result['LONGITUDE'] = te_predict[:,0]
    result.to_csv(result_csv_path,index=False)


if __name__=='__main__':
    start = time()
    main('result/result_mlp_0.1.csv',False)
    end = time()
    print('Time:\t'+str((end-start)/3600)+' hours')




