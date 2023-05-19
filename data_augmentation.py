import os
import os.path as osp
import time
import tsaug
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pylab as plt
import random
warnings.filterwarnings("ignore")

manual_seed = 2023
np.random.seed(manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
torch.random.manual_seed(manual_seed)
# DATA LOADER
# import pdb ; pdb.set_trace()
data_path = "./datasets/data/SWAT/ad_datasets_"
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = np.load(data_path + "/SWAT_train.npy")

scaler.fit(data)
data = scaler.transform(data)

import pandas as pd 
anomaly_ratio = 0.121
M = data.shape[0]
m=0 
l=0
anomaly_sequences = np.array([])
temp = np.array([])
while anomaly_sequences.sum()<=anomaly_ratio * M :
    anomaly_sequence = np.random.choice(np.arange(100,500),3)
    anomaly_sequences = np.concatenate([anomaly_sequences,anomaly_sequence])
num_anomaly_sequences = anomaly_sequences.shape[0]
start_points = np.random.choice(M,num_anomaly_sequences)
resample_points = np.random.choice(M,num_anomaly_sequences)
start_points = np.sort(start_points)
resample_points = np.sort(resample_points)

anomaly_ratio2 = 0.07
M = data.shape[0]
m=0 
l=0
anomaly_sequences2 = np.array([])
temp = np.array([])
while anomaly_sequences2.sum()<=anomaly_ratio2 * M :
    anomaly_sequence2 = np.random.choice(np.arange(100,500),3)
    anomaly_sequences2 = np.concatenate([anomaly_sequences2,anomaly_sequence2])
num_anomaly_sequences2 = anomaly_sequences2.shape[0]
start_points2 = np.random.choice(M,num_anomaly_sequences2)
start_points2 = np.sort(start_points2)

# Just resampling 
# import pdb ; pdb.set_trace()
Final_DATA = data.copy()
Final_label = np.zeros([data.shape[0]])
start_fix = start_points[0]
for i,start in enumerate(start_points):
    resample = resample_points[i]
    
    Final_1 = np.concatenate([Final_DATA[:start_fix],Final_DATA[resample:resample+int(anomaly_sequences[i])]])
    Final_2 = np.concatenate([Final_1,Final_DATA[start_fix:]])
    start_fix = start + int(anomaly_sequences[i])
    Final_DATA = Final_2.copy()
    
    Final_label_1 = np.concatenate([Final_label[:start_fix],np.ones([int(anomaly_sequences[i])])])
    Final_label_2 = np.concatenate([Final_label_1,Final_label[start_fix:]])
    
    Final_label = Final_label_2.copy()
# import pdb ; pdb.set_trace()

# for j,start2 in enumerate(start_points2):
#     sequences = anomaly_sequences2[j]
#     X3,Y3  = tsaug.Reverse().augment(np.arange(sequences), Final_DATA[start2: start2+int(sequences),0])
#     Final_DATA[start2:start2+int(sequences),0] = Y3
#     Final_label[start2:start2+int(sequences)] =1 

np.save('/home/bigdyl/PAD/PrecursorAnomalyDetection/datasets/data/SWAT_N/SWAT_SAMSUNG_train_label_mix.npy',Final_label)
np.save('/home/bigdyl/PAD/PrecursorAnomalyDetection/datasets/data/SWAT_N/SWAT_SAMSUNG_train_mix.npy',Final_DATA)   
