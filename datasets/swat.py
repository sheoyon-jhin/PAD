import collections as co
import numpy as np
import os
import pathlib

import torch
import urllib.request
import zipfile
import time_dataset
from . import common
import pandas as pd 
from sklearn import preprocessing
here = pathlib.Path(__file__).resolve().parent

DATA_PATH = os.path.dirname(os.path.abspath(__file__))

def _pad(channel, maxlen):
    # X중 하나의 데이터 들어옴 (Series) - (116,)
    channel = torch.tensor(channel) # Series를 tensor로 바꿈
    out = torch.full((maxlen,), channel[-1]) # tensor의 마지막 원소를 (maxlen,) 크기로 새로운 텐서 out을 만듦
    out[:channel.size(0)] = channel # 텐서 out의 원래 범위 만큼을 원래 값으로 채움
    return out # 리턴



def _process_data(look_window,forecast_window,stride_window,missing_rate,loc,learning_method):
    
    PATH = os.path.dirname(os.path.abspath(__file__))
    from sklearn import preprocessing
    torch.__version__
    # import pdb ; pdb.set_trace()
    data_path = PATH + '/data/SWAT_N'
    
    # # #Read data
    # normal = pd.read_csv(data_path + "/SWaT_Dataset_Normal_v1.csv",header=1)#, nrows=1000)
    # normal = normal.drop([" Timestamp" , "Normal/Attack" ] , axis = 1)
    
    # for i in list(normal): 
    #     normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
    # normal = normal.astype(float)
    
    # down_rate = 5 
    # normal=normal.groupby(np.arange(len(normal.index)) // down_rate).mean()
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x = normal.values
    # x_scaled = min_max_scaler.fit_transform(x)
    # normal = pd.DataFrame(x_scaled)
    # X_times = np.array(normal)
    # np.save(data_path+'/SWAT_train.npy',X_times)
    
    # attack = pd.read_csv(data_path + "/SWaT_Dataset_Attack_v0.csv",header=0)#, nrows=1000)
    # labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
    # attack = attack.drop([" Timestamp" , "Normal/Attack" ] , axis = 1)
    # for i in list(attack):
    #     attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
    
    # attack = attack.astype(float)
    # attack=attack.groupby(np.arange(len(attack.index)) // down_rate).mean()
    # #Downsampling the labels
    # labels_down=[]

    # for i in range(len(labels)//down_rate):
    #     if labels[5*i:5*(i+1)].count(1.0):
    #         labels_down.append(1.0) #Attack
    #     else:
    #         labels_down.append(0.0) #Normal
    
    # #for the last few labels that are not within a full-length window
    # if labels[down_rate*(i+1):].count(1.0):
    #     labels_down.append(1.0) #Attack
    # else:
    #     labels_down.append(0.0) #Normal
    # from sklearn import preprocessing

    # x = attack.values 
   
    # x_scaled = min_max_scaler.transform(x)
    # attack = pd.DataFrame(x_scaled)
    # X_times_test = np.array(attack)
    
    # test_labels = np.array(labels_down)
    # np.save(data_path+'/SWAT_test.npy',X_times_test)
    # np.save(data_path+'/SWAT_test_labels.npy',test_labels)
    if learning_method =='unsupervised':
        X_times = np.load(data_path + "/SWAT_train.npy")
    else:
        X_times = np.load(data_path + "/SWAT_train_mix.npy")
    # X_times = np.load(data_path + "/SWAT_train.npy")
    
    
    y_train_label = np.load(data_path + "/SWAT_train_label_mix.npy")
    X_times_test = np.load(data_path+"/SWAT_test.npy")
    y_label = np.load(data_path + '/SWAT_test_label.npy')
    input_size = X_times.shape[1]
    X_times = time_dataset.normalize(X_times)
    X_times_test = time_dataset.normalize(X_times_test)
    y_train_label = y_train_label[:X_times.shape[0]]
    total_length = len(X_times)
    
    timelen = X_times.shape[0]
    timelen_test = X_times_test.shape[0]
    
    full_seq_data_train = torch.Tensor()
    forecast_seq_train = torch.Tensor()
    full_y_seq_data_train = torch.Tensor()
    full_forecast_data_train = torch.Tensor()
    full_seq_data_test = torch.Tensor()
    full_y_seq_data = torch.Tensor()
    full_forecast_data_test = torch.Tensor()
    forecast_seq_test = torch.Tensor()
    
    # TRAIN 
    for _len in range(int((timelen-look_window+stride_window-forecast_window)/stride_window)):
        full_seq_temp = torch.Tensor([X_times[(_len*stride_window):(_len*stride_window)+look_window]])
        forecast_seq_temp = torch.Tensor([X_times[(_len*stride_window)+look_window:(_len*stride_window)+look_window+forecast_window]])

        full_y_seq_temp = torch.Tensor([y_train_label[(_len*stride_window):(_len*stride_window)+look_window]])
        forecast_y_seq_temp = torch.Tensor([y_train_label[(_len*stride_window)+look_window:(_len*stride_window)+look_window+forecast_window]])

        full_seq_data_train=torch.cat([full_seq_data_train,full_seq_temp])
        forecast_seq_train = torch.cat([forecast_seq_train,forecast_seq_temp])
        full_y_seq_data_train = torch.cat([full_y_seq_data_train,full_y_seq_temp])
        full_forecast_data_train = torch.cat([full_forecast_data_train,forecast_y_seq_temp])
    if learning_method =='unsupervised':
        if missing_rate ==0:
            
            DATA_PATH_SAVE =PATH + '/processed_data/UNS_EXPSWAT_508_look_'+str(look_window)+'_stride_'+str(stride_window)+'_forecast_'+str(forecast_window)
        else:
            DATA_PATH_SAVE =PATH + '/processed_data/UNS_EXPSWAT_508_look_'+str(look_window)+'_stride_'+str(stride_window)+'_forecast_'+str(forecast_window)+'_Missing_'+str(missing_rate)
        
    else:
        if missing_rate ==0:
            
            DATA_PATH_SAVE =PATH + '/processed_data/EXPSWAT_410_look_'+str(look_window)+'_stride_'+str(stride_window)+'_forecast_'+str(forecast_window)
        else:
            DATA_PATH_SAVE =PATH + '/processed_data/EXPSWAT_410_look_'+str(look_window)+'_stride_'+str(stride_window)+'_forecast_'+str(forecast_window)+'_Missing_'+str(missing_rate)
        
    # DATA_PATH_SAVE =PATH + '/processed_data/EXPSMD_121_look_'+str(look_window)+'_stride_'+str(stride_window)
    # full_y_data = torch.zeros([full_seq_data_train.shape[0],look_window])
    
    # import pdb ; pdb.set_trace()
    # TEST
    for _len in range(int((timelen_test-look_window+stride_window-forecast_window)/stride_window)):
        # import pdb ; pdb.set_trace()
        full_seq_temp = torch.Tensor([X_times_test[(_len*stride_window):(_len*stride_window)+look_window]])
        forecast_seq_temp = torch.Tensor([X_times_test[(_len*stride_window)+look_window:(_len*stride_window)+look_window+forecast_window]])
        
        full_y_seq_temp = torch.Tensor([y_label[(_len*stride_window):(_len*stride_window)+look_window]])
        forecast_y_seq_temp = torch.Tensor([y_label[(_len*stride_window)+look_window:(_len*stride_window)+look_window+forecast_window]])
        
        full_seq_data_test=torch.cat([full_seq_data_test,full_seq_temp])
        forecast_seq_test = torch.cat([forecast_seq_test,forecast_seq_temp])
        full_y_seq_data = torch.cat([full_y_seq_data,full_y_seq_temp])
        full_forecast_data_test = torch.cat([full_forecast_data_test,forecast_y_seq_temp])
    # import pdb ; pdb.set_trace()
    if missing_rate != 0:
        generator = torch.Generator().manual_seed(56789)
        # import pdb ; pdb.set_trace()
        for Xi in full_seq_data_train:
            removed_points = torch.randperm(full_seq_data_train.size(1), generator=generator)[:int(full_seq_data_train.size(1) * missing_rate)].sort().values
            Xi[removed_points] = float('nan')
        for Xi in full_seq_data_test:
            removed_points = torch.randperm(full_seq_data_test.size(1), generator=generator)[:int(full_seq_data_test.size(1) * missing_rate)].sort().values
            Xi[removed_points] = float('nan')
    
    eval_length = full_seq_data_test.shape[0]

    train_seq_data = full_seq_data_train
    train_y_data = full_y_seq_data_train
    train_forecast_data = full_forecast_data_train
    train_forecast_seq = forecast_seq_train
    
    val_seq_data = full_seq_data_test[:int(eval_length*0.5)]
    val_y_data = full_y_seq_data[:int(eval_length*0.5)]
    val_forecast_data = full_forecast_data_test[:int(eval_length*0.5)]
    val_forecast_seq = forecast_seq_test[:int(eval_length*0.5)]
    
    test_seq_data = full_seq_data_test[int(eval_length*0.5):]
    test_y_data = full_y_seq_data[int(eval_length*0.5):]
    test_forecast_data = full_forecast_data_test[int(eval_length*0.5):]
    test_forecast_seq = forecast_seq_test[int(eval_length*0.5):]
    # import pdb ; pdb.set_trace()
    torch.save(train_seq_data,DATA_PATH_SAVE+'/train_seq_data.pt')
    torch.save(train_y_data,DATA_PATH_SAVE+'/train_y_data.pt')
    torch.save(train_forecast_data,DATA_PATH_SAVE+'/train_forecast_y.pt')
    torch.save(train_forecast_seq,DATA_PATH_SAVE+'/train_forecast_seq.pt')
    

    torch.save(val_seq_data,DATA_PATH_SAVE+'/val_seq_data.pt')
    torch.save(val_y_data,DATA_PATH_SAVE+'/val_y_data.pt')
    torch.save(val_forecast_data,DATA_PATH_SAVE+'/val_forecast_y.pt')
    torch.save(val_forecast_seq,DATA_PATH_SAVE+'/val_forecast_seq.pt')

    torch.save(test_seq_data,DATA_PATH_SAVE+'/test_seq_data.pt')
    torch.save(test_y_data,DATA_PATH_SAVE+'/test_y_data.pt')
    torch.save(test_forecast_data,DATA_PATH_SAVE+'/test_forecast_y.pt')
    torch.save(test_forecast_seq,DATA_PATH_SAVE+'/test_forecast_seq.pt')

    times = torch.Tensor(np.arange(look_window))
    torch.save(times,DATA_PATH_SAVE+'/times.pt')
    return input_size

def get_data(look_window,forecast_window,stride_window,missing_rate,learning_method='self-supervised'):
    
    
    base_base_loc = here / 'processed_data'
    if learning_method=='unsupervised':
        if missing_rate ==0:
            loc = base_base_loc / ('UNS_EXPSWAT_508_look_'+str(look_window)+'_stride_'+str(stride_window)+'_forecast_'+str(forecast_window))
        # loc = base_base_loc / ('EXPSMD_121_look_'+str(look_window)+'_stride_'+str(stride_window)+'_Missing_'+str(missing_rate))
        else:
            loc = base_base_loc / ('UNS_EXPSWAT_508_look_'+str(look_window)+'_stride_'+str(stride_window)+'_forecast_'+str(forecast_window)+'_Missing_'+str(missing_rate))
    
    else:
        if missing_rate ==0:
            loc = base_base_loc / ('EXPSWAT_410_look_'+str(look_window)+'_stride_'+str(stride_window)+'_forecast_'+str(forecast_window))
        # loc = base_base_loc / ('EXPSMD_121_look_'+str(look_window)+'_stride_'+str(stride_window)+'_Missing_'+str(missing_rate))
        else:
            loc = base_base_loc / ('EXPSWAT_410_look_'+str(look_window)+'_stride_'+str(stride_window)+'_forecast_'+str(forecast_window)+'_Missing_'+str(missing_rate))
    if os.path.exists(loc):
        PATH = os.path.dirname(os.path.abspath(__file__))
        data_path = PATH + '/data/SWAT_N'
        # import pdb ; pdb.set_trace()
        X_times = np.load(data_path + "/SWAT_train_mix.npy")
        input_size = X_times.shape[1]
        # pass
    else:
        # download()
        
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        input_size = _process_data(look_window,forecast_window,stride_window,missing_rate,loc,learning_method)
        
    return loc,input_size