import numpy as np

import pytorch_msssim
import utils
from utils import standard_scaler, create_dataset, split_dataset, create_dataloader
from models import baseline_regressor, inverse_hypernet, forward_radiation,forward_GammaRad
import torch
import trainer
import matplotlib.pyplot as plt
import pickle
from losses import *

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train_paths,val_paths,test_paths = split_dataset()
    # paths_dict = {'train':train_paths,'val':val_paths,'test':test_paths}
    # pickle.dump(paths_dict,open('paths_dict_data_new.pkl','wb'))
    #create_dataset(train_paths,val_paths,test_paths)
    data = np.load(r'../AntennaDesign_data/data_dB.npz')
    train_params, train_gamma, train_radiation = data['parameters_train'], data['gamma_train'], data['radiation_train']
    val_params, val_gamma, val_radiation = data['parameters_val'], data['gamma_val'], data['radiation_val']
    test_params, test_gamma, test_radiation = data['parameters_test'], data['gamma_test'], data['radiation_test']
    scaler = standard_scaler()
    scaler.fit(train_params)
    train_params_scaled = scaler.forward(train_params)
    val_params_scaled = scaler.forward(val_params)
    test_params_scaled = scaler.forward(test_params)
    # ---------------------
    batch_sizes, epochs = [20], 70
    radiation_range = [-55,5]
    learning_rates = 0.0001
    step_sizes,gamma_schedule = [2],0.95
    inv_or_forw = 'inverse_forward_GammaRad'
    GammaRad_lambda,rad_phase_fac = 1,1 # GammaRad_lambda is the weight between radiation and gamma loss, rad_phase_fac is the weight between the phase and amplitude loss for radiation
    #----------------------
    training_losses = np.zeros((len(batch_sizes), len(step_sizes), epochs + 1))
    validation_losses = np.zeros((len(batch_sizes), len(step_sizes), epochs + 1))
    best_loss = 100
    best_dict = dict()
    for i, batch_size in enumerate(batch_sizes):
        for j, stp_size in enumerate(step_sizes):
            print('bs=', batch_size, ' step_size=', stp_size)
            model = inverse_hypernet.inverse_forward_concat(inv_module=inverse_hypernet.small_inverse_radiation_no_hyper(),
                                                            forw_module=forward_GammaRad.forward_GammaRad(rad_range=radiation_range),
                                                            forward_weights_path_rad=r'checkpoints/FORWARD_radiation_HuberCyclic_loss_range[-55,5].pth',
                                                            forward_weights_path_gamma=r'checkpoints/FORWARD_small_10layers_dB_linpred.pth')
            loss_fn = GammaRad_loss(lamda=GammaRad_lambda,rad_phase_fac=rad_phase_fac)
            # model.load_and_freeze_forward('FORWARD_small_gamma_loss_10layers.pth')
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates)
            train_loader = create_dataloader(train_gamma, train_radiation, train_params_scaled, batch_size, device,inv_or_forw)
            val_loader = create_dataloader(val_gamma, val_radiation, val_params_scaled, batch_size, device,inv_or_forw)
            test_loader = create_dataloader(test_gamma, test_radiation, test_params_scaled, batch_size, device,inv_or_forw)
            print(f'seccessfully created data loaders for {inv_or_forw} training')
            train_los, train_los_stds, val_los, val_los_stds,test_loss = trainer.run_model(model, loss_fn, optimizer,
                                                                                 train_loader, val_loader, test_loader,
                                                                                 epochs, stp_size, gamma_schedule,inv_or_forw)

            training_losses[i, j] = np.array(train_los)
            validation_losses[i, j] = np.array(val_los)
            if test_loss < best_loss:
                best_model = model
                best_loss = test_loss
                best_dict['bs'] = batch_size
                best_dict['step_size'] = stp_size
            plt.figure()
            plt.plot(np.arange(len(val_los)), val_los, label='validation')
            plt.plot(np.arange(len(val_los)), train_los, label='train')
            #plt.plot(np.arange(len(val_los)), np.ones(len(val_los)) * nn_loss.item(), label='NN loss', color='k')
            plt.title(f'best model test loss = {test_loss:.3f}')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.figure()
            plt.plot(np.arange(len(val_los)), val_los_stds, label='validation')
            plt.plot(np.arange(len(val_los)), train_los_stds, label='train')
            plt.title(f'data')
            plt.xlabel('epoch')
            plt.ylabel('std')
            plt.legend()
            print('saving best model')
            torch.save(model.state_dict(), f'checkpoints/INVERSE_GammaRad_concat_HuberCyclic_loss.pth')
    plt.show()