import scipy.io as sio
import numpy as np

import utils
from utils import standard_scaler, create_dataset, split_dataset
from models import baseline_regressor, inverse_hypernet
import torch
import trainer
import matplotlib.pyplot as plt


def create_dataloader(gamma, radiation, params_scaled, batch_size, device):
    gamma = torch.tensor(gamma).to(device).float()
    radiation = torch.tensor(radiation).to(device).float()
    params_scaled = torch.tensor(params_scaled).to(device).float()
    dataset = torch.utils.data.TensorDataset(gamma, radiation, params_scaled)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train_paths,val_paths,test_paths = split_dataset()
    # create_dataset(train_paths,val_paths,test_paths)
    data = np.load('data.npz')
    train_params, train_gamma, train_radiation = data['parameters_train'], data['gamma_train'], data['radiation_train']
    val_params, val_gamma, val_radiation = data['parameters_val'], data['gamma_val'], data['radiation_val']
    test_params, test_gamma, test_radiation = data['parameters_test'], data['gamma_test'], data['radiation_test']
    scaler = standard_scaler()
    scaler.fit(train_params)
    train_params_scaled = torch.tensor(scaler.forward(train_params)).to(device).float()
    val_params_scaled = torch.tensor(scaler.forward(val_params)).to(device).float()
    test_params_scaled = torch.tensor(scaler.forward(test_params)).to(device).float()

    batch_size, epochs = 32, 65
    loss_fn = torch.nn.MSELoss()
    learning_rates = [0.001]
    step_sizes = [15]

    training_losses = np.zeros((len(learning_rates), len(step_sizes), epochs + 1))
    validation_losses = np.zeros((len(learning_rates), len(step_sizes), epochs + 1))
    best_loss = 100
    best_dict = dict()
    for i, lr in enumerate(learning_rates):
        for j, stp_size in enumerate(step_sizes):
            print('lr=', lr, ' step_size=', stp_size)
            model = inverse_hypernet.inverse_radiation_no_hyper()
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            train_loader = create_dataloader(train_gamma, train_radiation, train_params_scaled, batch_size, device)
            val_loader = create_dataloader(val_gamma, val_radiation, val_params_scaled, batch_size, device)
            test_loader = create_dataloader(test_gamma, test_radiation, test_params_scaled, batch_size, device)

            # nn_loss = utils.NN_benchmark(loss_fn,val_gamma,val_params_scaled)
            train_los, train_los_stds, val_los, val_los_stds = trainer.run_model(model, loss_fn, optimizer,
                                                                                 train_loader, val_loader, test_loader,
                                                                                 epochs, batch_size, stp_size, device)

            training_losses[i, j] = np.array(train_los)
            validation_losses[i, j] = np.array(val_los)
            if val_los[-1] < best_loss:
                best_model = model
                best_loss = val_los[-1]
                best_dict['lr'] = lr
                best_dict['step_size'] = stp_size
            plt.figure()
            plt.plot(np.arange(len(val_los)), val_los, label='validation - inverse')
            plt.plot(np.arange(len(val_los)), train_los, label='train - inverse')
            plt.plot(np.arange(len(val_los)), np.ones(len(val_los)) * nn_loss.item(), label='NN loss', color='k')
            plt.title('inverse network - avg loss vs epochs (with dropout,3 added layers model size)')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.figure()
            plt.plot(np.arange(len(val_los)), val_los_stds, label='validation - inverse')
            plt.plot(np.arange(len(val_los)), train_los_stds, label='train - inverse')
            plt.title('inverse network -loss std vs epochs (with dropout,3 added layers model size) ')
            plt.xlabel('epoch')
            plt.ylabel('std')
            plt.legend()
        plt.show()
