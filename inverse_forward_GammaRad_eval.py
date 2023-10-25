import matplotlib.pyplot as plt
from utils import *
import trainer
from models import baseline_regressor, inverse_hypernet, forward_radiation,forward_GammaRad
from losses import *
import torch
import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'../AntennaDesign_data/data_dB.npz')
    parser.add_argument('--model_path', type=str, default='checkpoints/INVERSE_GammaRad_concat_HuberCyclic_loss_lamda1_radphasefac1_lr0.0002_loss0.674.pth')
    parser.add_argument('--inv_or_forw', type=str, default='inverse_forward_GammaRad',
    help='architecture name, to parse dataset correctly. options: inverse, forward_gamma, forward_radiation, inverse_forward_gamma, inverse_forward_GammaRad')
    parser.add_argument('--rad_range', type=list, default=[-55,5], help='range of radiation values [dB] for scaling')
    parser.add_argument('--sample', type=int, default=17, help='sample to plot its output, from test set')

    return parser.parse_args()

def main():
    args = arg_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = np.load(args.data_path)
    train_params, train_gamma, train_radiation = data['parameters_train'], data['gamma_train'], data['radiation_train']
    val_params, val_gamma, val_radiation = data['parameters_val'], data['gamma_val'], data['radiation_val']
    test_params, test_gamma, test_radiation = data['parameters_test'], data['gamma_test'], data['radiation_test']
    radiation_range = args.rad_range

    model = inverse_hypernet.inverse_forward_concat(inv_module=inverse_hypernet.small_inverse_radiation_no_hyper(),
                                                    forw_module=forward_GammaRad.forward_GammaRad(rad_range=radiation_range))
    model.to(device)
    model.load_state_dict(torch.load(args.model_path,map_location=device))
    inv_or_forw = args.inv_or_forw
    scaler = standard_scaler()
    scaler.fit(train_params)
    train_params_scaled = scaler.forward(train_params)
    val_params_scaled = scaler.forward(val_params)
    test_params_scaled = scaler.forward(test_params)
    batch_size = val_gamma.shape[0]
    test_loader = create_dataloader(val_gamma, val_radiation, val_params_scaled, batch_size, device, inv_or_forw)
    predicted_spectrum,gt_spectrum = trainer.evaluate_model(model, None, test_loader, 'test', inv_or_forw, return_output=True)
    test_loader = create_dataloader(val_gamma, val_radiation, val_params_scaled, batch_size, device, 'inverse')
    predicted_geo, gt_geo = trainer.evaluate_model(model.inverse_module,None,test_loader,'test','inverse',return_output=True)
    #---
    pred_gamma, pred_radiation = predicted_spectrum
    GT_gamma, GT_radiation = gt_spectrum
    produce_stats_gamma(gt_spectrum, predicted_spectrum,'dB')
    produce_radiation_stats(gt_spectrum, predicted_spectrum)
    prnt = inv_or_forw
    sample = args.sample
    pred_gamma_sample = pred_gamma[sample].cpu().detach().numpy()
    pred_gamma_sample[:int(0.5 * GT_gamma.shape[1])] = 10*np.log10(pred_gamma_sample[:int(0.5 * GT_gamma.shape[1])] )
    GT_gamma_sample = GT_gamma[sample].cpu().detach().numpy()
    plt.figure()
    plt.imshow(pred_radiation[sample,2,:,:].cpu().detach().numpy())
    plt.title(f'Predicted radiation pattern phase, {prnt} loss, sample {sample}')
    plt.colorbar()
    plt.figure()
    plt.imshow(GT_radiation[sample,2,:,:].cpu().detach().numpy())
    plt.title(f'Ground truth radiation pattern phase, sample {sample}')
    plt.colorbar()
    plt.figure()
    plt.imshow(pred_radiation[sample,0,:,:].cpu().detach().numpy())
    plt.title(f'Predicted radiation pattern magnitude, {prnt} loss, sample {sample}')
    plt.colorbar()
    plt.figure()
    plt.imshow(GT_radiation[sample,0,:,:].cpu().detach().numpy())
    plt.title(f'Ground truth radiation pattern magnitude, sample {sample}')
    plt.colorbar()
    plt.figure()
    plt.plot(pred_gamma_sample,label='Predicted gamma')
    plt.plot(GT_gamma_sample,label='Ground truth gamma')
    plt.plot(np.ones(20) * 0.5 * GT_gamma.shape[1], np.arange(-1, 1, 0.1), 'k--')
    plt.title(f'Gamma, {prnt} loss, sample {sample}')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()