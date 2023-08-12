import scipy.io as sio
from os import listdir
from os.path import isfile, join
import numpy as np

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
    num_of_test_points = num_of_data_points - num_of_train_points - num_of_val_points

    train_pick = np.random.choice(num_of_data_points, num_of_train_points, replace=False)
    val_pick = np.random.choice(np.setdiff1d(np.arange(num_of_data_points), train_pick), num_of_val_points, replace=False)
    test_pick = np.setdiff1d(np.arange(num_of_data_points), np.concatenate((train_pick, val_pick)), assume_unique=True)

    train_dataset_path_list = dataset_path_list[train_pick]
    val_dataset_path_list = dataset_path_list[val_pick]
    test_dataset_path_list = dataset_path_list[test_pick]
    return train_dataset_path_list, val_dataset_path_list, test_dataset_path_list







