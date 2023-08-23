import torch
import torch.nn as nn
from models import resnet


class inverse_radiation_no_hyper(nn.Module):
    def __init__(self):
        super(inverse_radiation_no_hyper,self).__init__()
        self.relu = nn.ELU()
        self.radiation_backbone = nn.Sequential(resnet.ResNetBasicBlock(4,16),resnet.ResNetBasicBlock(16,16),resnet.ResNetBasicBlock(16,16))

    def forward(self,input):
        gamma, radiation = input
        if radiation.ndim == 3:
            radiation = radiation.unsqueeze(0)
        radiation_features = self.radiation_backbone(radiation)
        return torch.ones((32,12))



