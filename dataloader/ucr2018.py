# -*- coding: utf-8 -*-
import numpy as np
import torch.utils.data as data
import random


class UCR2018(data.Dataset):

    def __init__(self, data, targets, transform):
        self.data = np.asarray(data, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            if random.random() < 0.5:
                img = np.flip(img, axis=1).copy()
                target = target[::-1].copy()
        return img, target

    def __len__(self):
        return self.data.shape[0]


def load_ucr2018(dataset_path, dataset_name, nb_class):
    ##################
    # load raw data
    ##################
    print("nb_class:\n",nb_class)
    print("dataset_name:\n",dataset_name)


    if dataset_name in['Beibuwan']:
        data_des = "dataset"
        print("data_des:\n",data_des)
        data = np.load("{}/{}/{}.npy".format(dataset_path, dataset_name, data_des))
        
        print("data_shape:\n",data.shape)
        np.random.shuffle(data)
        train_size = int(data.shape[0]*0.8)
        val_size = int(data.shape[0]*0.9)
        (x_train, x_val, x_test)=(data[:train_size, :, :-1], data[train_size:val_size, :, :-1], data[val_size:, :,  :-1])
        (y_train, y_val, y_test)=(data[:train_size, :, -1:], data[train_size:val_size, :,-1:], data[val_size:, :,  -1:])


    nb_timesteps = 144
    input_shape = (nb_timesteps, 6)

    ########################################
    # Data Split End
    ########################################

    # Process data

    x_test = x_test.reshape((-1, input_shape[0], input_shape[1]))
    x_val = x_val.reshape((-1, input_shape[0], input_shape[1]))
    x_train = x_train.reshape((-1, input_shape[0], input_shape[1]))

    print("Train:{}, Test:{}, Class(For each timestep):{}".format(x_train.shape, x_test.shape, nb_class))

    # # Normalize
    # x_train_max = np.max(x_train)
    # x_train_min = np.min(x_train)

    # x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
    # # Test is secret
    # x_val = 2. * (x_val - x_train_min) / (x_train_max - x_train_min) - 1.
    # x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.

    return x_train, y_train, x_val, y_val, x_test, y_test,data_des #class_stat_all