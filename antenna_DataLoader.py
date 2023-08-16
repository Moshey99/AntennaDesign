import scipy.io as sio
import numpy as np

import utils
from utils import standard_scaler, create_dataset , split_dataset
from models import baseline_regressor
import torch
import trainer
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_paths,val_paths,test_paths = split_dataset()
    create_dataset(train_paths,val_paths,test_paths)
    data = np.load('data.npz')
    train_params,train_gamma = data['parameters_train'],data['gamma_train']
    #utils.display_gamma(train_gamma[100])
    val_params,val_gamma = data['parameters_val'],data['gamma_val']
    test_params,test_gamma = data['parameters_test'],data['gamma_test']
    scaler = standard_scaler()
    scaler.fit(train_params)
    train_params_scaled = torch.tensor(scaler.forward(train_params)).to(device)
    val_params_scaled = torch.tensor(scaler.forward(val_params)).to(device)
    test_params_scaled = torch.tensor(scaler.forward(test_params)).to(device)
    model = baseline_regressor.baseline_inverse_model()
    model.to(device)

    batch_size,epochs = 32,30
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_gamma, val_gamma, test_gamma = torch.tensor(train_gamma).to(device), torch.tensor(val_gamma).to(device), torch.tensor(test_gamma).to(device)
    nn_loss = utils.NN_benchmark(loss_fn, val_gamma, val_params_scaled)

    trainer.run_model(model,loss_fn,optimizer,(train_gamma,train_params_scaled),(val_gamma,val_params_scaled),
                      (test_gamma,test_params_scaled),epochs,batch_size,device)









