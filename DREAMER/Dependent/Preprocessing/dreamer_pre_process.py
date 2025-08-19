import os

import scipy

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import scipy.io as sio
import argparse
import sys
import numpy as np
import pandas as pd
import time
import pickle
import re

np.random.seed(0)

def data_1Dto2D(data, Y=9, X=9):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0,  	   	0, 	        0,          0,          0,          0, 	        0,  	    0, 	        0       )
    data_2D[1] = (0,  	   	0,          0,          data[0],    0,          data[13],   0,          0,          0       )
    data_2D[2] = (data[1],  0,          0,          0,          0,          0,          data[11],   0,          data[12])
    data_2D[3] = (0,        data[3],    0,          0,          0,          0,          0,          data[10],   0       )
    data_2D[4] = (data[4],  0,          data[2],    0,          0,          0,          0,          0,          data[9] )
    data_2D[5] = (0,        0,          0,          0,          0,          0,          0,          0,          0       )
    data_2D[6] = (data[5],  0,          0,          0,          0,          0,          0,          0,          data[8] )
    data_2D[7] = (0,        0,          0,          0,          0,          0,          0,          0,          0       )
    data_2D[8] = (0,        0,          0,          data[6],    0,          data[7],    0,          0,          0       )
    # return shape:9*9
    return data_2D

def norm_dataset(dataset_1D):
    norm_dataset_1D = np.zeros([dataset_1D.shape[0], dataset_1D.shape[-1]])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
    # return shape: m*32
    return norm_dataset_1D

def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data. nonzero ()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
    # return shape: 9*9
    return data_normalized

def dataset_1Dto2D(dataset_1D):
    dataset_2D = np.zeros([dataset_1D.shape[0],9,9])
    for i in range(dataset_1D.shape[0]):
        dataset_2D[i] = data_1Dto2D(dataset_1D[i])
    # return shape: m*9*9
    return dataset_2D

def norm_dataset_1Dto2D(dataset_1D):
    norm_dataset_2D = np.zeros([dataset_1D.shape[0], 9, 9])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_2D[i] = feature_normalize( data_1Dto2D(dataset_1D[i]))
    return norm_dataset_2D

def windows(data, size):
    start = 0
    while ((start+size) < data.shape[0]):
        yield int(start), int(start + size)
        start += size

def segment_signal(data, label, label_index, window_size):
    # get data file name and label file name
    for (start, end) in windows(data, window_size):
        # print(data.shape)
        if ((len(data[start:end]) == window_size)):
            if (start == 0):
                segments = data[start:end]
                segments = np.vstack([segments, data[start:end]])

                labels = np.array(label[label_index])
                labels = np.append(labels, np.array(label[label_index]))
            else:
                segments = np.vstack([segments, data[start:end]])
                labels = np.append(labels, np.array(label[label_index]))  # labels = np.append(labels, stats.mode(label[start:end])[0][0])
    return segments, labels

def segment_baseline(base, window_size):
    # get data file name
    for (start, end) in windows(base, window_size):
        # print(data.shape)
        if ((len(base[start:end]) == window_size)):
            if (start == 0):
                segments_ = base[start:end]
                segments_ = np.vstack([segments_, base[start:end]])
            else:
                segments_ = np.vstack([segments_, base[start:end]])
    return segments_

if __name__ == '__main__':
    begin = time.time()

    print("time begin:", time.localtime())
    window_size = 128
    all_data_eeg = np.empty([0, window_size, 14])
    all_data_ecg = np.empty([0, window_size, 2])
    all_label_arousal = np.empty([0])
    all_label_valence = np.empty([0])
    all_label_four = np.empty([0])

    label_class_arousal = "arousal"
    label_class_valence = "valence"  # sys.argv[1]
    label_class_four = "four"  # sys.argv[1]     # arousal/valence/dominance
    # arousal/valence/dominance
    eeg = []
    ecg =[]
    label_v = []
    label_a = []
    label_f = []

    # sys.argv[1]     # arousal/valence/dominance
    subs = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']                # sys.argv[2]
    for sub in subs:

        debase = 'yes'
        dataset_dir = '/media/gdlls/My Book/Root/cuikai/dataset/Dreamer/'

        dataset_dir1 =dataset_dir+'stimuli(m,14)/'+sub+'_stimuli'
        data_file1 = sio.loadmat(dataset_dir1 + ".mat")
        dataset_dir1_ecg =dataset_dir+'stimuli_ecg/'+sub+'_stimuli'
        data_file1_ecg = sio.loadmat(dataset_dir1_ecg + ".mat")

        dataset_dir2_arousal =dataset_dir+'label_'+label_class_arousal+'(18,1)/'+sub+'_'+label_class_arousal+'_label'  #arousal/valence/dominance
        arousal = sio.loadmat(dataset_dir2_arousal + ".mat")

        dataset_dir2_valence = dataset_dir + 'label_' + label_class_valence + '(18,1)/' + sub + '_' + label_class_valence + '_label'  # arousal/valence/dominance
        valence = sio.loadmat(dataset_dir2_valence + ".mat")

        dataset_dir3 =dataset_dir+'baseline(m,14)/'+sub+'_baseline'  #
        data_file3 = sio.loadmat(dataset_dir3 + ".mat")
        dataset_dir3_ecg = dataset_dir + 'baseline_ecg/' + sub + '_baseline'  #
        data_file3_ecg = sio.loadmat(dataset_dir3_ecg + ".mat")

        output_dir = '../Data_processing/DREAMER_Pre/'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)



        # load label
        label_in_arousal =arousal['label']
        label_in_valence =valence['label']
        label_four_sub = np.zeros_like(label_in_valence)
        label_four_sub[(label_in_arousal >= 3) & (label_in_valence >= 3)] = 0
        label_four_sub[(label_in_arousal >= 3) & (label_in_valence < 3)] = 1
        label_four_sub[(label_in_arousal < 3) & (label_in_valence >= 3)] = 2
        label_four_sub[(label_in_arousal < 3) & (label_in_valence < 3)] = 3

        label_arousal_sub = [1 if x >= 3 else 0 for x in label_in_arousal]
        label_valence_sub = [1 if x >= 3 else 0 for x in label_in_valence]




        label_inter_arousal = np.empty([0])
        label_inter_valence = np.empty([0])
        label_inter_four = np.empty([0])

        # data_inter_cnn = np.empty([0,window_size, 9, 9])
        data_inter_rnn = np.empty([0, window_size, 14])
        baseline_inter = np.empty([0, window_size, 14])
        base_signal_ = np.zeros([window_size, 14])
        base_signal = np.empty([18, window_size, 14])
        # data_inter_cnn_ecg = np.empty([0,window_size, 9, 9])
        data_inter_rnn_ecg = np.empty([0, window_size, 2])
        baseline_inter_ecg = np.empty([0, window_size, 2])
        base_signal__ecg = np.zeros([window_size, 2])
        base_signal_ecg = np.empty([18, window_size, 2])
        # load dataset
        dataset_in = data_file1
        dataset_in_ecg = data_file1_ecg

        key_list = [key for key in data_file1.keys() if key.startswith('__') == False]
        print(key_list)
        key_list_rearrange = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for key in key_list:
            index = int(re.findall('\d+', key)[0]) - 1
            key_list_rearrange[index] = key
        print(key_list_rearrange)

        key_list2 = [key for key in data_file3.keys() if key.startswith('__') == False]
        print(key_list2)
        key_list_rearrange2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for key in key_list2:
            index = int(re.findall('\d+', key)[0]) - 1
            key_list_rearrange2[index] = key
        print(key_list_rearrange2)

        for key in key_list_rearrange2:
            if key.startswith('__') == False:  # if array is EEG than do as follow
                print("Processing ",  key, "..........")
                baseline = data_file3[key]
                baseline_ecg = data_file3_ecg[key]
                label_index = int(re.findall('\d+', key)[0]) - 1  # get number from str key
                # data = norm_dataset(data)  # normalization
                print('shape of this EEG: ', baseline.shape)
                baseline = segment_baseline(baseline, window_size)
                baseline_ecg = segment_baseline(baseline_ecg, window_size)

                base_segment_set = baseline.reshape(int(baseline.shape[0] / window_size), window_size, 14)
                base_segment_set_ecg = baseline_ecg.reshape(int(baseline_ecg.shape[0] / window_size), window_size, 2)

                print('segment number of this EEG: ', base_segment_set.shape[0])
                baseline_inter = np.vstack([baseline_inter, base_segment_set])
                baseline_inter_ecg = np.vstack([baseline_inter_ecg, base_segment_set_ecg])

                print(baseline_inter.shape)

        for k in range(1, 19):
            for i in range(1, 62):
                base_signal_ += baseline_inter[k * i - 1]
                base_signal__ecg += baseline_inter_ecg[k * i - 1]

            base_signal_ = base_signal_ / 61
            base_signal[k-1] = base_signal_
            base_signal__ecg = base_signal__ecg / 61
            base_signal_ecg[k - 1] = base_signal__ecg
        print(base_signal.shape)

        count = 1
        # traversing 18 EEGs of one experiment/session
        data_eeg_t = []
        data_ecg_t = []
        label_v_t = []
        label_a_t = []
        label_f_t = []
        for key in key_list_rearrange:
             if key.startswith('__') == False:  # if array is EEG than do as follow
              print("Processing ", count, key, "..........")
              count = count + 1
              data = dataset_in[key][:7680]
              data_ecg = dataset_in_ecg[key][:7680]

              label_index = int(re.findall('\d+', key)[0]) - 1  # get number from str key

              if debase =='yes':
                  for m in range(0, data.shape[0]//128):
                      data[m * window_size:(m + 1) * window_size, 0:14] = data[m * window_size:(m + 1) * window_size, 0:14] - base_signal[label_index]
                      data_ecg[m * window_size:(m + 1) * window_size, 0:2] = data[m * window_size:(m + 1) * window_size, 0:2] - base_signal_ecg[label_index]

              data_sub = norm_dataset(data)  # normalization
              data_sub_ecg = norm_dataset(data_ecg)  # normalization

              print('shape of this EEG: ', data.shape)
              data, label_arousal = segment_signal(data_sub, label_arousal_sub, label_index, window_size)
              _, label_valence = segment_signal(data_sub, label_valence_sub, label_index, window_size)
              data_ecg, label_four = segment_signal(data_sub_ecg, label_four_sub, label_index, window_size)

              # # cnn data process
              # data_cnn = dataset_1Dto2D(data)
              # data_cnn = data_cnn.reshape(int(data_cnn.shape[0] / window_size), window_size, 9, 9)
              # rnn data process
              data_rnn = data.reshape(int(data.shape[0] / window_size), window_size, 14)
              data_rnn_ecg = data_ecg.reshape(int(data_ecg.shape[0] / window_size), window_size, 2)

              data_eeg_t.append(data_rnn)
              data_ecg_t.append(data_rnn_ecg)
              label_a_t.append(label_arousal)
              label_v_t.append(label_valence)
              label_f_t.append(label_four)

        data_eeg_t = np.array(data_eeg_t)
        data_ecg_t = np.array(data_ecg_t)
        label_a_t = np.array(label_a_t)
        label_v_t = np.array(label_v_t)
        label_f_t = np.array(label_f_t)

        eeg.append(data_eeg_t)
        ecg.append(data_ecg_t)
        label_a.append(label_a_t)
        label_v.append(label_v_t)
        label_f.append(label_f_t)
        end = time.time()
        print("end time:", time.asctime(time.localtime(time.time())))
        print("time consuming:", (end - begin))
    eeg = np.array(eeg)
    ecg = np.array(ecg)
    label_a = np.array(label_a)
    label_v = np.array(label_v)
    label_f = np.array(label_f)
    eeg = eeg.transpose(1, 2, 0, 4, 3)
    ecg = ecg.transpose(1, 2, 0, 4, 3)
    label_a = np.transpose(label_a, (1, 2, 0))
    label_v = np.transpose(label_v, (1, 2, 0))
    label_f = np.transpose(label_f, (1, 2, 0))

    eeg = np.reshape(eeg, (-1, eeg.shape[2], 1, eeg.shape[-2], eeg.shape[-1]))
    ecg = np.reshape(ecg, (-1, ecg.shape[2], 1, ecg.shape[-2], ecg.shape[-1]))
    label_a = np.reshape(label_a, (-1, label_a.shape[-1]))
    label_v = np.reshape(label_v, (-1, label_v.shape[-1]))
    label_f = np.reshape(label_f, (-1, label_f.shape[-1]))

    eeg_dir = output_dir + "all_dataset_eeg.pkl"
    ecg_dir = output_dir + "all_dataset_ecg.pkl"
    label_arousal_dir = output_dir + 'all_' + label_class_arousal + '_labels.pkl'
    label_valence_dir = output_dir + 'all_' + label_class_valence + '_labels.pkl'
    label_four_dir = output_dir + 'all_' + label_class_four + '_labels.pkl'


    with open(eeg_dir, "wb") as fp:
        pickle.dump(eeg, fp, protocol=4)
    with open(ecg_dir, "wb") as fp:
        pickle.dump(ecg, fp, protocol=4)
    with open(label_arousal_dir, "wb") as fp:
        pickle.dump(label_a, fp)
    with open(label_valence_dir, "wb") as fp:
        pickle.dump(label_v, fp)
    with open(label_four_dir, "wb") as fp:
        pickle.dump(label_f, fp)