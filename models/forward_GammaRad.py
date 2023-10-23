import torch
import torch.nn as nn
from models import baseline_regressor,forward_radiation
class forward_GammaRad(nn.Module):
    def __init__(self,gamma_model=None,radiation_model=None,rad_range=[-55,5]):
        super(forward_GammaRad,self).__init__()
        if gamma_model is None:
            self.gamma_net = baseline_regressor.small_deeper_baseline_forward_model()
        else:
            self.gamma_net = gamma_model

        if radiation_model is None:
            self.radiation_net = forward_radiation.Radiation_Generator(rad_range)
        else:
            self.radiation_net = radiation_model
    def forward(self,x):
        gamma_pred = self.gamma_net(x)
        radiation_pred = self.radiation_net(x)
        return gamma_pred,radiation_pred