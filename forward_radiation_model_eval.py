import matplotlib.pyplot as plt
from utils import *
import trainer
from models import baseline_regressor, inverse_hypernet, forward_radiation
from losses import *
import torch



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = np.load(r'../AntennaDesign_data/data_dB.npz')
    train_params, train_gamma, train_radiation = data['parameters_train'], data['gamma_train'], data['radiation_train']
    val_params, val_gamma, val_radiation = data['parameters_val'], data['gamma_val'], data['radiation_val']
    test_params, test_gamma, test_radiation = data['parameters_test'], data['gamma_test'], data['radiation_test']
    model = forward_radiation.Radiation_Generator()
    lamda = [0,1]
    radiation_range = [-55,5]
    for lm in lamda:
        loss_fn = radiation_loss_dB()
        if lm == 0:
            loss_fn.dB_magnitude_loss = pytorch_msssim.MSSSIM(radiation_range=radiation_range)
            prnt = 'MSSSIM'
        else:
            loss_fn.dB_magnitude_loss = nn.HuberLoss()
            prnt = 'Huber'
        model.load_state_dict(torch.load(f'checkpoints/FORWARD_radiation_{prnt}Cyclic_loss_range[{radiation_range[0]},{radiation_range[1]}].pth'))
        inv_or_forw = 'forward_radiation'
        scaler = standard_scaler()
        scaler.fit(train_params)
        train_params_scaled = scaler.forward(train_params)
        test_params_scaled = scaler.forward(test_params)
        batch_size = test_gamma.shape[0]
        test_loader = create_dataloader(test_gamma, test_radiation, test_params_scaled, batch_size, device, inv_or_forw)
        pred_radiation = trainer.evaluate_model(model, loss_fn, test_loader, 'test', inv_or_forw, return_output=True)
        GT_radiation =  test_loader.dataset.tensors[1]
        produce_radiation_stats(pred_radiation,GT_radiation)
        # sample = 10
        # plt.figure()
        # plt.imshow(pred_radiation[sample,2,:,:].cpu().detach().numpy())
        # plt.title(f'Predicted radiation pattern phase, {prnt} loss')
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(GT_radiation[sample,2,:,:].cpu().detach().numpy())
        # plt.title(f'Ground truth radiation pattern phase')
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(pred_radiation[sample,0,:,:].cpu().detach().numpy())
        # plt.title(f'Predicted radiation pattern magnitude, {prnt} loss')
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(GT_radiation[sample,0,:,:].cpu().detach().numpy())
        # plt.title(f'Ground truth radiation pattern magnitude')
        # plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()