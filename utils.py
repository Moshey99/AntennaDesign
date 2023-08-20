import scipy.io as sio
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def split_dataset(dataset_path= r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data', train_val_test_split = [0.6,0.2, 0.2]):
    folders = listdir(dataset_path)
    dataset_path_list = []

    for folder in folders:
        folder_path = join(dataset_path, folder)
        files = listdir(folder_path)
        for file in files:
            file_path = join(folder_path, file)
            if isfile(file_path):
                if file.endswith('.mat') and file.__contains__('results'):
                    dataset_path_list.append(file_path)

    dataset_path_list = np.array(dataset_path_list)
    num_of_data_points = len(dataset_path_list)
    num_of_train_points = int(num_of_data_points * train_val_test_split[0])
    num_of_val_points = int(num_of_data_points * train_val_test_split[1])

    train_pick = np.random.choice(num_of_data_points, num_of_train_points, replace=False)
    val_pick = np.random.choice(np.setdiff1d(np.arange(num_of_data_points), train_pick), num_of_val_points, replace=False)
    test_pick = np.setdiff1d(np.arange(num_of_data_points), np.concatenate((train_pick, val_pick)), assume_unique=True)

    train_dataset_path_list = dataset_path_list[train_pick]
    val_dataset_path_list = dataset_path_list[val_pick]
    test_dataset_path_list = dataset_path_list[test_pick]
    return train_dataset_path_list, val_dataset_path_list, test_dataset_path_list
def create_dataset(dataset_path_list_train,dataset_path_list_val,dataset_path_list_test):
    print('Creating dataset...')
    data_parameters_train,data_parameters_val,data_parameters_test = [],[],[]
    data_gamma_train,data_gamma_val,data_gamma_test = [],[],[]
    for path in dataset_path_list_train:
        mat = sio.loadmat(path)
        parameters = np.squeeze(mat['parameters'])
        if path.__contains__('V1'):
            parameters = np.concatenate((parameters,np.array([0,0,19.55])))
        gamma = np.squeeze(mat['gamma'])
        gamma = np.concatenate((np.abs(gamma),np.angle(gamma)))
        data_parameters_train.append(parameters)
        data_gamma_train.append(gamma)
    for path in dataset_path_list_val:
        mat = sio.loadmat(path)
        parameters = np.squeeze(mat['parameters'])
        if path.__contains__('V1'):
            parameters = np.concatenate((parameters,np.array([0,0,19.55])))
        gamma = np.squeeze(mat['gamma'])
        gamma = np.concatenate((np.abs(gamma),np.angle(gamma)))
        data_parameters_val.append(parameters)
        data_gamma_val.append(gamma)
    for path in dataset_path_list_test:
        mat = sio.loadmat(path)
        parameters = np.squeeze(mat['parameters'])
        if path.__contains__('V1'):
            parameters = np.concatenate((parameters,np.array([0,0,19.55])))
        gamma = np.squeeze(mat['gamma'])
        gamma = np.concatenate((np.abs(gamma),np.angle(gamma)))
        data_parameters_test.append(parameters)
        data_gamma_test.append(gamma)

    np.savez('data.npz',parameters_train=np.array(data_parameters_train),gamma_train=np.array(data_gamma_train),
             parameters_val=np.array(data_parameters_val),gamma_val=np.array(data_gamma_val),
             parameters_test=np.array(data_parameters_test),gamma_test=np.array(data_gamma_test))
    print('Dataset created seccessfully. Saved in data.npz')
class standard_scaler():
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self,data):
        self.mean = np.mean(data,axis=0)
        self.std = np.std(data,axis=0)
    def forward(self,input):
        return (input - self.mean)/self.std
    def inverse(self,input):
        return input*self.std + self.mean

def NN_benchmark(loss,x_array,y_array):
    avg_loss = 0
    for i in range(x_array.shape[0]):
        x = x_array[i]
        tmp_x_array = torch.clone(x_array)
        tmp_x_array[i] = 100
        nearest_x_idx = torch.sum(torch.abs(tmp_x_array-x),dim=1).argmin()
        tmp_loss = loss(y_array[i],y_array[nearest_x_idx])
        avg_loss += tmp_loss
    avg_loss /= x_array.shape[0]
    print('NN benchmark loss: ',avg_loss)
    return avg_loss

def display_gamma(gamma):

    gamma_real = gamma[:int(gamma.shape[0]/2)]
    gamma_imag = gamma[int(gamma.shape[0]/2):]
    gamma_mag = np.sqrt(gamma_real**2 + gamma_imag**2)
    gamma_phase = np.arctan2(gamma_imag,gamma_real)
    plt.figure()
    plt.plot(np.arange(len(gamma_mag)),gamma_mag)
    plt.show()
def display_losses(train_loss,val_loss):
    plt.figure()
    plt.plot(np.arange(len(train_loss)),train_loss,label='train loss')
    plt.plot(np.arange(len(val_loss)),val_loss,label='val loss')
    plt.legend()
    plt.show()
