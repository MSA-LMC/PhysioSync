"""
This code is used to perform data process for DEAP dataset.
"""
import scipy.io as sio
import argparse
import os
import numpy as np
import pandas as pd
import time
import pickle

np.random.seed(0)
#Print process
def print_top(dataset_dir, window_size):#, convert, parallel, segment, begin_subject, end_subject, output_dir, set_store):
    print("######################## DEAP EEG data preprocess ########################	\
    		   \n# input directory:	%s \
    		   \n# window size:		%d 	\
    		   \n##############################################################################" % \
          (dataset_dir, window_size))
    return None

#Norm_dataset function
def norm_dataset(dataset_1D):
    print("---------------------------------------------------------------------------------------------------------------")
    print("dataset-1D:",dataset_1D.shape)
    norm_dataset_1D = np.zeros([dataset_1D.shape[0], 32])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
    return norm_dataset_1D

#Feature_normalize
def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data.nonzero()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
    return data_normalized

#function of getting EEG windon
def windows(data, size):
    start = 0
    while ((start + size) <= data.shape[0]):
        yield int(start), int(start + size)
        start += size

#segment_signal_without_transition
def segment_signal_without_transition(data,label,label_index,window_size):
    print(" label index:", label_index)
    print("data shape:", data.shape)
    for (start, end) in windows(data, window_size):
       # break
        if ((len(data[start:end]) == window_size)):
            if(start == 0):
                segments = data[start:end]
                labels = np.array(label[label_index])
                print('label:', labels)
            else:
                segments = np.vstack([segments, data[start:end]])
                labels = np.append(labels, np.array(label[label_index]))
    print(set(labels))
    return segments, labels


#Process function
def process(dataset_file, window_size):
    data_file_in = pickle.load(open(file,'rb'), encoding='latin1')

    data_in = data_file_in["data"].transpose(0,2,1) #(40, 8064, 40)(trial channel data)
    #0 valence, 1 arousal, 2 dominance, 3 liking
    if dimension == "A":
        label_in = data_file_in["labels"][:,1]>=5 # select label
    elif dimension == "V":
        label_in = data_file_in["labels"][:,0]>=5
    elif dimension == "Four":
        valence = data_file_in["labels"][:, 0]
        arouse = data_file_in["labels"][:, 1]

        label_in = np.zeros_like(arouse)

        label_in[(arouse >= 5) & (valence >= 5)] = 0
        label_in[(arouse >= 5) & (valence < 5)] = 1
        label_in[(arouse < 5) & (valence >= 5)] = 2
        label_in[(arouse <= 5) & (valence <= 5)] = 3

    label_inter = []
    data_inter = []
    trials = data_in.shape[0]

#Process for each trial
    for trial in range(0, trials):
        base_signal = (data_in[trial,0:128,0:32] + data_in[trial,128:256,0:32]+ data_in[trial,256:384,0:32])/3  # baseline signal is the average of the first 3s
        data = data_in[trial,384:8064,0:32]
        # compute the deviation between baseline signals and experimental signals
        for i in range(0, 60):
            data[i*128:(i+1)*128,0:32] = data[i*128:(i+1)*128,0:32] - base_signal
        label_index = trial
        print("trial:", trial)
        print("data shape:", data.shape)
        data = norm_dataset(data)
        data, label = segment_signal_without_transition(data, label_in, label_index, window_size)
        #rnn
        data = data.reshape(int(data.shape[0]/window_size), window_size, 32)
        data_inter.append(data)
        label_inter.append(label)

    data_inter = np.array(data_inter)
    label_inter = np.array(label_inter)
    print("total data size:", data_inter.shape)
    print("total label size:", label_inter.shape)
    return data_inter,label_inter  # record

dimension = "A"  # A V Four
window = "5s" # 1s 5s

dataset_dir = '/media/gdlls/My Book/Root/cuikai/dataset/DEAP/data_preprocessed_python' #root of raw data
save_dir = "../Data_processing/DEAP_" + window + "/" + dimension + "/"
os.makedirs(save_dir, exist_ok=True)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

DATA_DEAP = []
LABEL_DEAP = []

if window == "1s":
    window_size = 128
elif window == "5s":
    window_size = 640

#Process for each subjects
record_list = [task for task in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, task))]
print(record_list)
for record in record_list:
    file = os.path.join(dataset_dir, record)
    print_top(file, window_size)
    data_inter,label_inter = process(file, window_size)
    DATA_DEAP.append(data_inter)
    LABEL_DEAP.append(label_inter)

DATA_DEAP = np.array(DATA_DEAP)
DATA_DEAP = np.transpose(DATA_DEAP, (1, 2, 0, 4, 3))
LABEL_DEAP = np.array(LABEL_DEAP)
LABEL_DEAP = np.transpose(LABEL_DEAP, (1, 2, 0))
from sklearn.model_selection import train_test_split
DATA_DEAP = np.reshape(DATA_DEAP, (-1,DATA_DEAP.shape[2], 1,DATA_DEAP.shape[-2],DATA_DEAP.shape[-1]))
LABEL_DEAP = np.reshape(LABEL_DEAP, (-1, LABEL_DEAP.shape[-1]))

np.save(os.path.join(save_dir, 'DATA_DEAP.npy'), DATA_DEAP)
np.save(os.path.join(save_dir, 'LABEL_DEAP.npy'), LABEL_DEAP)

