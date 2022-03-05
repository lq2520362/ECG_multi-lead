#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

import numpy as np
import scipy.io as scio
import csv
import os
from data import QRSDetectorOffline


def data_pro(data_path):
    ECG = []
    Reference_index = []
    # max_data = 17.881176
    # min_data = -14.625054
    # normalization_max = 3
    # normalization_min = -3
    for file in os.listdir(data_path):
        if not file.endswith('.mat'):
            continue
        file_path = os.path.join(data_path, file)
        print('processing ===> ' + file_path)
        Normal = scio.loadmat(file_path)
        Normal = np.array(Normal['ECG'][0][0][2])

        # z-score normalization
        # for j in range(12):
        #     Normal[j] = (Normal[j] - np.mean(Normal[j])) / np.std(Normal[j])
        # # # min-max normalization
        # for j in range(12):
        #     Normal[j] = normalization_min + (normalization_max - normalization_min) * (Normal[j] - min_data) / (
        #                 max_data - min_data)

        signal = np.transpose(Normal)
        print(signal.shape)
        qrsdetector = QRSDetectorOffline.QRSDetectorOffline(signal, 500, verbose=False,
                                                            plot_data=False, show_plot=False)
        for i in range(signal.shape[1]):
            signal[:, i] = qrsdetector.bandpass_filter(signal[:, i], lowcut=0.5, highcut=49.0,
                                                       signal_freq=500, filter_order=1)

        # len_signal = signal.shape[0]
        # if len_signal > 500 * 60:
        #     len_signal = 500 * 60
        # if len_signal > 3000:
        #     index = 10
        #     Reference_index.append(index)
        #     len_st = int((len_signal - 3000) / (index - 1))
        #     for i in range(index):
        #         signal_arr = signal[(i * len_st):(i * len_st + 3000), :]
        #         signal_arr = np.transpose(signal_arr)
        #         ECG.append(signal_arr)

        len_signal = signal.shape[0]
        if len_signal > 3000:
            index = int((len_signal / 3000) + 1)
            Reference_index.append(index)
            len_pad = math.ceil((index * 3000 - len_signal) / (index - 1))
            for i in range(index):
                signal_arr = signal[(i * (3000 - len_pad)):(i * (3000 - len_pad) + 3000), :]
                signal_arr = np.transpose(signal_arr)
                ECG.append(signal_arr)

        else:
            Reference_index.append(1)
            signal = np.transpose(signal)
            ECG.append(signal)

            # len_signal_add = 3000 - len_signal
            # signal_add = signal[:len_signal_add, :]
            # signal = np.concatenate((signal, signal_add), axis=0)
            # signal = np.transpose(signal)
            # ECG.append(signal)

        # zero_len = 3600 - signal.shape[0]
        # if zero_len > 0:
        #     zero = np.zeros((zero_len, 12))
        #     signal = np.concatenate((signal, zero), axis=0)
        # else:
        #     signal = signal[:3600, :]
        # signal = np.transpose(signal)
        # ECG.append(signal)
    print(len(ECG))
    np.save('ECG_train_data_21445.npy', ECG)

    # ----label----
    with open('TrainingSet/REFERENCE.csv', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        column = [row[1] for row in reader]

    ECG_label = []
    for i in range(len(column) - 1):
        for j in range(Reference_index[i]):
            ECG_label.append(int(column[i + 1]) - 1)

    ECG_label_arr = np.array(ECG_label)
    print(ECG_label_arr.shape)

    np.save('ECG_train_label_21445.npy', ECG_label_arr)

    return ECG, ECG_label


if __name__ == '__main__':
    data_path = 'TrainingSet'
    Data, Label = data_pro(data_path)

    # data_split(Data, Label)

    # ==read data==
    # store = pd.HDFStore('Train_Test/ECG_data_float32.h5')
    # df1 = store['data']
    # mat = np.array(df1)
    # print(mat.shape)
    # store.close()

    # data = pd.read_hdf('Train_Test/ECG_data_float32.h5')
    # # df = pd.DataFrame(data)
    # print(data)

    print('Program completed !!!')
