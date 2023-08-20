import scipy.io as sio
import numpy as np

import utils
from utils import standard_scaler, create_dataset , split_dataset
from models import baseline_regressor
import torch
import trainer
import matplotlib.pyplot as plt
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train_paths,val_paths,test_paths = split_dataset()
    # create_dataset(train_paths,val_paths,test_paths)
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
    batch_size,epochs = 32,2
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.SmoothL1Loss()
    learning_rates = [0.001]
    step_sizes = [15]
    training_losses = np.zeros((len(learning_rates),len(step_sizes),epochs+1))
    validation_losses = np.zeros((len(learning_rates),len(step_sizes),epochs+1))
    best_loss = 100
    best_dict = dict()
    for i,lr in enumerate(learning_rates):
        for j,stp_size in enumerate(step_sizes):
          print('lr=',lr,' step_size=',stp_size)
          model = baseline_regressor.baseline_forward_model()
          model.to(device)
          optimizer = torch.optim.Adam(model.parameters(), lr=lr)
          train_gamma, val_gamma, test_gamma = torch.tensor(train_gamma).to(device), torch.tensor(val_gamma).to(device), torch.tensor(test_gamma).to(device)
          #nn_loss = utils.NN_benchmark(loss_fn, val_params_scaled,val_gamma)

          train_los,train_los_std,val_los,val_stds = trainer.run_model(model,loss_fn,optimizer,(train_params_scaled,train_gamma),(val_params_scaled,val_gamma),
                                                (test_params_scaled,test_gamma),epochs,batch_size,stp_size,device)
          training_losses[i,j] = np.array(train_los)
          validation_losses[i,j] = np.array(val_los)
          if val_los[-1]<best_loss:
            best_model = model
            best_loss = val_los[-1]
            best_dict['lr']=lr
            best_dict['step_size']=stp_size

          plt.figure(1)
          plt.plot(np.arange(1,epochs+1),training_losses[i,j],label=f'lr={lr} step={stp_size}')
          plt.title('Training Loss')
          plt.legend()
          plt.figure(2)
          plt.plot(np.arange(1,epochs+1),validation_losses[i,j],label=f'lr={lr} step={stp_size}')
          plt.title('Validation Loss')
          plt.legend()
    plt.show()
    torch.save(best_model.state_dict(),'forward_baseline_model_nolayer7.pt')












