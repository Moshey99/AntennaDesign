import scipy.io as sio
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import time
import trainer
import losses
from models import baseline_regressor, inverse_hypernet
import random

def create_dataloader(gamma, radiation, params_scaled, batch_size, device,inv_or_forw = 'inverse'):
    gamma = torch.tensor(gamma).to(device).float()
    radiation = torch.tensor(radiation).to(device).float()
    params_scaled = torch.tensor(params_scaled).to(device).float()
    if inv_or_forw == 'inverse':
        dataset = torch.utils.data.TensorDataset(gamma, radiation, params_scaled)
    elif inv_or_forw == 'forward':
        dataset = torch.utils.data.TensorDataset(params_scaled, gamma)
    elif inv_or_forw == 'inverse_forward':
        dataset = torch.utils.data.TensorDataset(gamma, radiation)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader
def nearest_neighbor_loss(loss_fn,x_train,y_train, x_val, y_val, k=1):
    strt = time.time()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x_train)
    distances, indices = nbrs.kneighbors(x_val)
    cnt = len(np.where(distances < 0.1)[0])
    nearest_neighbor_y = y_train[indices].squeeze()
    loss = loss_fn(torch.tensor(nearest_neighbor_y), torch.tensor(y_val))
    return loss

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
    if train_val_test_split[2] > 0:
        test_pick = np.setdiff1d(np.arange(num_of_data_points), np.concatenate((train_pick, val_pick)), assume_unique=True)
    else:
        test_pick = val_pick

    train_dataset_path_list = dataset_path_list[train_pick]
    val_dataset_path_list = dataset_path_list[val_pick]
    test_dataset_path_list = dataset_path_list[test_pick]
    return train_dataset_path_list, val_dataset_path_list, test_dataset_path_list
def create_dataset(dataset_path_list_train,dataset_path_list_val,dataset_path_list_test):
    print('Creating dataset...')
    data_parameters_train,data_parameters_val,data_parameters_test = [],[],[]
    data_gamma_train,data_gamma_val,data_gamma_test = [],[],[]
    data_radiation_train,data_radiation_val,data_radiation_test = [],[],[]

    for path in dataset_path_list_train:
        mat = sio.loadmat(path)
        parameters = np.squeeze(mat['parameters'])
        if path.__contains__('V1'):
            parameters = np.concatenate((parameters,np.array([0,0,19.55])))
        gamma = np.squeeze(mat['gamma'])
        gamma = np.concatenate((np.abs(gamma),np.angle(gamma)))
        rad = np.squeeze(mat['farfield'])[:,:,1:,0]
        rad_concat = np.concatenate((np.abs(rad),np.angle(rad)),axis=2)
        rad_concat_swapped = np.swapaxes(rad_concat,0,2)
        data_radiation_train.append(rad_concat_swapped)
        data_parameters_train.append(parameters)
        data_gamma_train.append(gamma)

    for path in dataset_path_list_val:
        mat = sio.loadmat(path)
        parameters = np.squeeze(mat['parameters'])
        if path.__contains__('V1'):
            parameters = np.concatenate((parameters,np.array([0,0,19.55])))
        gamma = np.squeeze(mat['gamma'])
        gamma = np.concatenate((np.abs(gamma),np.angle(gamma)))
        rad = np.squeeze(mat['farfield'])[:,:,1:,0]
        rad_concat = np.concatenate((np.abs(rad),np.angle(rad)),axis=2)
        rad_concat_swapped = np.swapaxes(rad_concat,0,2)
        data_radiation_val.append(rad_concat_swapped)
        data_parameters_val.append(parameters)
        data_gamma_val.append(gamma)

    for path in dataset_path_list_test:
        mat = sio.loadmat(path)
        parameters = np.squeeze(mat['parameters'])
        if path.__contains__('V1'):
            parameters = np.concatenate((parameters,np.array([0,0,19.55])))
        gamma = np.squeeze(mat['gamma'])
        gamma = np.concatenate((np.abs(gamma),np.angle(gamma)))
        rad = np.squeeze(mat['farfield'])[:,:,1:,0]
        rad_concat = np.concatenate((np.abs(rad),np.angle(rad)),axis=2)
        rad_concat_swapped = np.swapaxes(rad_concat,0,2)
        data_radiation_test.append(rad_concat_swapped)
        data_parameters_test.append(parameters)
        data_gamma_test.append(gamma)

    np.savez('data_new.npz',parameters_train=np.array(data_parameters_train),gamma_train=np.array(data_gamma_train),
             radiation_train=np.array(data_radiation_train),parameters_val=np.array(data_parameters_val),
             gamma_val=np.array(data_gamma_val),radiation_val=np.array(data_radiation_val),parameters_test=np.array(data_parameters_test),
             gamma_test=np.array(data_gamma_test),radiation_test=np.array(data_radiation_test))
    print('Dataset created seccessfully. Saved in data_new.npz')
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
def display_gradients_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
def downsample_gamma(gamma,rate):
    gamma_len = gamma.shape[1]
    gamma_mag = gamma[:,:int(gamma_len/2)]
    gamma_phase = gamma[:,int(gamma_len/2):]
    gamma_mag_downsampled = gamma_mag[:,::rate]
    gamma_phase_downsampled = gamma_phase[:,::rate]
    gamma_downsampled = np.concatenate((gamma_mag_downsampled,gamma_phase_downsampled),axis=1)
    return gamma_downsampled
def downsample_radiation(radiation,rates):
    first_dim_rate, second_dim_rate = rates
    radiation_downsampled = radiation[:,:,::first_dim_rate,::second_dim_rate]
    return radiation_downsampled

def convert_dataset_to_dB(data):
    train_params, train_gamma, train_radiation = data['parameters_train'], data['gamma_train'], data['radiation_train']
    val_params, val_gamma, val_radiation = data['parameters_val'], data['gamma_val'], data['radiation_val']
    test_params, test_gamma, test_radiation = data['parameters_test'], data['gamma_test'], data['radiation_test']
    train_gamma[:,:int(train_gamma.shape[1]/2)] = 10*np.log10(train_gamma[:,:int(train_gamma.shape[1]/2)])
    val_gamma[:,:int(val_gamma.shape[1]/2)] = 10*np.log10(val_gamma[:,:int(val_gamma.shape[1]/2)])
    test_gamma[:,:int(test_gamma.shape[1]/2)] = 10*np.log10(test_gamma[:,:int(test_gamma.shape[1]/2)])
    train_radiation[:,:int(train_radiation.shape[1]/2)] = 10*np.log10(train_radiation[:,:int(train_radiation.shape[1]/2)])
    val_radiation[:,:int(val_radiation.shape[1]/2)] = 10*np.log10(val_radiation[:,:int(val_radiation.shape[1]/2)])
    test_radiation[:,:int(test_radiation.shape[1]/2)] = 10*np.log10(test_radiation[:,:int(test_radiation.shape[1]/2)])
    np.savez('data_dB.npz',parameters_train=train_params,gamma_train=train_gamma,radiation_train=train_radiation,
                parameters_val=val_params,gamma_val=val_gamma,radiation_val=val_radiation,
                parameters_test=test_params,gamma_test=test_gamma,radiation_test=test_radiation)
    print('Dataset converted to dB. Saved in data_dB.npz')

def reorganize_data(data):
    features_to_exclude = [5,6]
    train_params, train_gamma, train_radiation = data['parameters_train'], data['gamma_train'], data['radiation_train']
    val_params, val_gamma, val_radiation = data['parameters_val'], data['gamma_val'], data['radiation_val']
    test_params, test_gamma, test_radiation = data['parameters_test'], data['gamma_test'], data['radiation_test']
    all_params, all_gamma, all_radiation = np.concatenate((train_params,val_params,test_params),axis=0), np.concatenate((train_gamma,val_gamma,test_gamma),axis=0), np.concatenate((train_radiation,val_radiation,test_radiation),axis=0)
    first_feature,second_feature = all_params[:,features_to_exclude[0]], all_params[:,features_to_exclude[1]]
    pct20first, pct30first = np.percentile(first_feature,20), np.percentile(first_feature,30)
    pct70second, pct80second =  np.percentile(second_feature,60), np.percentile(second_feature,80)
    test_params_idx = np.where(np.logical_or(np.logical_and(first_feature>pct20first,first_feature<pct30first),
                                         np.logical_and(second_feature>pct70second,second_feature<pct80second)))
    test_params_new, test_gamma_new, test_radiation_new = all_params[test_params_idx], all_gamma[test_params_idx], all_radiation[test_params_idx]
    train_params_new, train_gamma_new, train_radiation_new = np.delete(all_params,test_params_idx,axis=0), np.delete(all_gamma,test_params_idx,axis=0), np.delete(all_radiation,test_params_idx,axis=0)
    val_idx = np.random.choice(train_params_new.shape[0],int(0.25*train_params_new.shape[0]),replace=False) # 25% of remaining data is about 20% of original data
    val_params_new, val_gamma_new, val_radiation_new = train_params_new[val_idx], train_gamma_new[val_idx], train_radiation_new[val_idx]
    train_params_new, train_gamma_new, train_radiation_new = np.delete(train_params_new,val_idx,axis=0), np.delete(train_gamma_new,val_idx,axis=0), np.delete(train_radiation_new,val_idx,axis=0)
    np.savez('data_reorganized.npz',parameters_train=train_params_new,gamma_train=train_gamma_new,radiation_train=train_radiation_new,
                parameters_val=val_params_new,gamma_val=val_gamma_new,radiation_val=val_radiation_new,
                parameters_test=test_params_new,gamma_test=test_gamma_new,radiation_test=test_radiation_new)

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = np.load(r'../AntennaDesign_data/data.npz')
    train_params, train_gamma, train_radiation = data['parameters_train'], data['gamma_train'], data['radiation_train']
    val_params, val_gamma, val_radiation = data['parameters_val'], data['gamma_val'], data['radiation_val']
    test_params, test_gamma, test_radiation = data['parameters_test'], data['gamma_test'], data['radiation_test']
    #--- for small model ---
    train_gamma = downsample_gamma(train_gamma,4)
    val_gamma = downsample_gamma(val_gamma,4)
    test_gamma = downsample_gamma(test_gamma,4)
    #--- for small model ---
    scaler = standard_scaler()
    scaler.fit(train_params)
    train_params_scaled = scaler.forward(train_params)
    val_params_scaled = scaler.forward(val_params)
    test_params_scaled = scaler.forward(test_params)
    loss = losses.gamma_loss()
    nn_loss_backward = nearest_neighbor_loss(loss,train_gamma,train_params_scaled,test_gamma,test_params_scaled)
    nn_loss_forward = nearest_neighbor_loss(loss,train_params_scaled,train_gamma,val_params_scaled,val_gamma)
    print('NN loss backward: ',nn_loss_backward.item())
    print('NN loss forward: ',nn_loss_forward.item())
    inverse_net_concat = inverse_hypernet.inverse_forward_concat()
    inverse_net_concat.inverse_module.load_state_dict(torch.load('INVERSE_small_gammasloss_forward10layers.pth'))
    inverse_net_concat.forward_module.load_state_dict(torch.load('FORWARD_small_gamma_loss_10layers.pth'))
    test_loader = create_dataloader(test_gamma,test_radiation,test_params_scaled,test_gamma.shape[0],device,inv_or_forw='inverse_forward')
    test_gamma_output = torch.load('output_gamma_concat.pth')
    sample = 60
    test_gamma_output_sample = test_gamma_output[sample]
    plt.plot(np.arange(len(test_gamma_output_sample)), test_gamma_output_sample, 'b', label='reconstructed gamma')
    plt.plot(np.arange(len(test_gamma[sample])),test_gamma[sample],'r',label='true gamma')
    plt.plot(np.ones(20)*0.5*test_gamma.shape[1],np.arange(-1,1,0.1),'k--')
    plt.title(' GT gamma VS reconstructed gamma, sample #'+str(sample))
    geometry_loss = nn.HuberLoss()
    test_loader_inverse = create_dataloader(test_gamma, test_radiation, test_params_scaled, 1, device,inv_or_forw='inverse')
    trainer.evaluate_model(inverse_net_concat.inverse_module,geometry_loss,test_loader_inverse,'test','inverse')
    plt.legend()
    plt.show()