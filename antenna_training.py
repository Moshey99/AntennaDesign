import scipy.io as sio
import numpy as np
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetRegressor
import utils
from utils import standard_scaler, create_dataset, split_dataset
from models import baseline_regressor, inverse_hypernet
import torch
import trainer
import matplotlib.pyplot as plt
import pickle

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
    # paths_dict = {'train':train_paths,'val':val_paths,'test':test_paths}
    # pickle.dump(paths_dict,open('paths_dict_data_new.pkl','wb'))
    #create_dataset(train_paths,val_paths,test_paths)
    data = np.load('data.npz')
    train_params, train_gamma, train_radiation = data['parameters_train'], data['gamma_train'], data['radiation_train']
    val_params, val_gamma, val_radiation = data['parameters_val'], data['gamma_val'], data['radiation_val']
    test_params, test_gamma, test_radiation = data['parameters_test'], data['gamma_test'], data['radiation_test']
    scaler = standard_scaler()
    scaler.fit(train_params)
    train_params_scaled = scaler.forward(train_params)
    val_params_scaled = scaler.forward(val_params)
    test_params_scaled = scaler.forward(test_params)

    batch_sizes, epochs = [1], 40
    loss_fn = torch.nn.MSELoss()
    learning_rates = 0.0001
    step_sizes,gamma_schedule = [2],0.96

    training_losses = np.zeros((len(batch_sizes), len(step_sizes), epochs + 1))
    validation_losses = np.zeros((len(batch_sizes), len(step_sizes), epochs + 1))
    best_loss = 100
    best_dict = dict()
    for i, batch_size in enumerate(batch_sizes):
        for j, stp_size in enumerate(step_sizes):
            print('bs=', batch_size, ' step_size=', stp_size)
            model = inverse_hypernet.inverse_radiation_hyper()
            # model = baseline_regressor.baseline_inverse_model()
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates)
            train_loader = create_dataloader(train_gamma, train_radiation, train_params_scaled, batch_size, device)
            val_loader = create_dataloader(val_gamma, val_radiation, val_params_scaled, batch_size, device)
            test_loader = create_dataloader(test_gamma, test_radiation, test_params_scaled, batch_size, device)
            print('seccessfully created data loaders')
            # nn_loss = utils.NN_benchmark(loss_fn,val_gamma,val_params_scaled)
            train_los, train_los_stds, val_los, val_los_stds,test_loss = trainer.run_model(model, loss_fn, optimizer,
                                                                                 train_loader, val_loader, test_loader,
                                                                                 epochs, stp_size, gamma_schedule,grad_accumulation_step=5)

            training_losses[i, j] = np.array(train_los)
            validation_losses[i, j] = np.array(val_los)
            if test_loss < best_loss:
                best_model = model
                best_loss = test_loss
                best_dict['bs'] = batch_size
                best_dict['step_size'] = stp_size
            plt.figure()
            plt.plot(np.arange(len(val_los)), val_los, label='validation - inverse')
            plt.plot(np.arange(len(val_los)), train_los, label='train - inverse')
            #plt.plot(np.arange(len(val_los)), np.ones(len(val_los)) * nn_loss.item(), label='NN loss', color='k')
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
            log_dict = {
                'model_name': 'inverse_model_radiation_resnet_no_hyper',
                'batch_size': batch_size,
                'lr': learning_rates,
                'epochs': epochs,
                'step_size scheduler': best_dict['step_size'],
                'loss_type': 'MSE',
                'loss_value': best_loss,
                'data_split': [60, 20, 20]}
            import pickle

            model_name = log_dict['model_name']
            with open(f'log_dictionary_{model_name}.pkl', 'wb') as f:
                pickle.dump(log_dict, f)
    print('saving best model')
    torch.save(best_model.state_dict(), f'best_model_{model_name}.pth')
