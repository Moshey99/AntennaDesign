import torch
import torch.nn as nn
from models import resnet


class inverse_radiation_no_hyper(nn.Module):
    def __init__(self,p_drop=0.25):
        super(inverse_radiation_no_hyper,self).__init__()
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(p=p_drop)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.radiation_backbone = nn.Sequential(resnet.ResNetBasicBlock(4,16),resnet.ResNetBasicBlock(16,16),resnet.ResNetBasicBlock(16,16),resnet.ResNetBasicBlock(16,16),
                                                nn.Conv2d(16,32,kernel_size=3),nn.BatchNorm2d(32),self.relu,self.maxpool,
                                                nn.Conv2d(32,128,kernel_size=3),nn.BatchNorm2d(128),self.relu,self.maxpool,
                                                nn.Conv2d(128,512,kernel_size=3),nn.BatchNorm2d(512),self.relu,self.maxpool,
                                                nn.Conv2d(512,1024,kernel_size=3),nn.BatchNorm2d(1024),self.relu,self.maxpool,
                                                nn.Conv2d(1024,2002,kernel_size=3),nn.BatchNorm2d(2002),self.relu,nn.MaxPool2d(kernel_size=(7,1),stride=1))
        self.fc1 = nn.Linear(4004,1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024,1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024,128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128,12)



    def forward(self,input):
        gamma, radiation = input
        if radiation.ndim == 3:
            radiation = radiation.unsqueeze(0)
        radiation_features = self.radiation_backbone(radiation)
        radiation_features = radiation_features.view(radiation_features.shape[0],-1)
        x = torch.cat((gamma,radiation_features),dim=1)
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.fc4(x)

        return x

class inverse_radiation_hyper(nn.Module):
    def __init__(self, weight_range=0.1, p_drop=0.25):
        super(inverse_radiation_hyper,self).__init__()
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(p=p_drop)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.hyper_fc_shapes = [(435,12)]
        self.radiation_backbone = nn.Sequential(resnet.ResNetBasicBlock(4,16),resnet.ResNetBasicBlock(16,16),resnet.ResNetBasicBlock(16,16),resnet.ResNetBasicBlock(16,16),
                                                nn.Conv2d(16,32,kernel_size=3),nn.BatchNorm2d(32),self.relu,self.maxpool,
                                                nn.Conv2d(32,128,kernel_size=3),nn.BatchNorm2d(128),self.relu,self.maxpool,
                                                nn.Conv2d(128,256,kernel_size=3),nn.BatchNorm2d(256),self.relu,self.maxpool,
                                                nn.Conv2d(256,512,kernel_size=3),nn.BatchNorm2d(512),self.relu,self.maxpool,
                                                nn.Conv2d(512,750,kernel_size=3))
        self.fc1 = nn.Linear(2002,800)
        self.fc2 = nn.Linear(800,435)
        self.init_weights(weight_range)

    def init_weights(self,init_range):
        for p in self.parameters():
            p.data.uniform_(-init_range, init_range)

    def forward(self,input):
        gamma, radiation = input
        radiation_features = self.radiation_backbone(radiation)
        radiation_features = radiation_features.view(-1)
        x = self.relu(self.fc1(gamma))
        x = self.relu(self.fc2(x))
        total_parameters = 0
        for i,fc_shape in enumerate(self.hyper_fc_shapes):
            in_size,out_size = fc_shape
            cnt = in_size*out_size + out_size*2
            total_parameters += cnt
            weights = radiation_features[:in_size*out_size].view(out_size,in_size)
            scale = radiation_features[in_size*out_size:in_size*out_size+out_size]
            bias = radiation_features[in_size*out_size+out_size:in_size*out_size+out_size*2]
        if i == len(self.hyper_fc_shapes)-1:
            x = self.scaled_linear(weights,scale,bias,input=x)
        else:
            x = self.scaled_linear(weights,scale,bias,input=x)
            x = self.relu(x)

        return x


    def scaled_linear(self,weights,scale,bias,input):
        input = input.matmul(weights.t())*scale + bias
        return input


if __name__ == "__main__":
    model = inverse_radiation_hyper()
    # gamma = torch.randn(1,2002)
    # radiation = torch.randn(1,4,91,181)
    # output = model((gamma,radiation))
    #print model parameters amount
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
