import torch
import torch.nn as nn

class baseline_forward_model(nn.Module):
    """
    model designed to find the regression between 12 geometric parameters as input and the spectrum parameters as output.
    for now spectrum parameters are 1001x2 = 2002 parameters.
    """
    def __init__(self):
        super(baseline_forward_model,self).__init__()
        self.linear1 = nn.Linear(12, 30)
        self.linear2 = nn.Linear(30, 60)
        self.linear3 = nn.Linear(60, 120)
        self.linear4 = nn.Linear(120, 240)
        self.linear5 = nn.Linear(240, 240)
        self.linear6 = nn.Linear(240, 480)
        self.linear7 = nn.Linear(480, 480)
        self.linear8 = nn.Linear(480, 960)
        self.linear9 = nn.Linear(960, 960)
        self.linear10 = nn.Linear(960, 2002)
        self.relu = nn.ELU()
    def forward(self,input):
        output = self.relu(self.linear1(input))
        output = self.relu(self.linear2(output))
        output = self.relu(self.linear3(output))
        output = self.relu(self.linear4(output))
        output = self.relu(self.linear5(output))
        output = self.relu(self.linear6(output))
        output = self.relu(self.linear7(output))
        output = self.relu(self.linear8(output))
        output = self.relu(self.linear9(output))
        output = self.linear10(output) # no activation function at the end, that is the geometric parameters
        return output

class baseline_inverse_model(nn.Module):
    """
    model designed to find the regression between 2002 spectrum parameters as input and the 12 geometric parameters as output.
    """
    def __init__(self):
        super(baseline_inverse_model,self).__init__()
        self.linear1 = nn.Linear(2002, 960)
        self.linear2 = nn.Linear(960, 960)
        self.linear3 = nn.Linear(960, 480)
        self.linear4 = nn.Linear(480, 480)
        self.linear5 = nn.Linear(480, 240)
        self.linear6 = nn.Linear(240, 240)
        self.linear7 = nn.Linear(240, 120)
        self.linear8 = nn.Linear(120, 60)
        self.linear9 = nn.Linear(60, 30)
        self.linear10 = nn.Linear(30, 12)
        self.relu = nn.ELU()
    def forward(self,input):
        output = self.relu(self.linear1(input))
        output = self.relu(self.linear2(output))
        output = self.relu(self.linear3(output))
        output = self.relu(self.linear4(output))
        output = self.relu(self.linear5(output))
        output = self.relu(self.linear6(output))
        output = self.relu(self.linear7(output))
        output = self.relu(self.linear8(output))
        output = self.relu(self.linear9(output))
        output = self.linear10(output) # no activation function at the end, that is the geometric parameters
        return output
class baseline_inverse_forward_model(nn.Module):
    """
    model designed to return the input spectrum parameters as output spectrum parameters.
    """
    def __init__(self):
        super(baseline_inverse_forward_model,self).__init__()
        self.inverse_part = baseline_inverse_model()
        self.forward_part = baseline_forward_model()
    def forward(self,input):
        geometric_parameters = self.inverse_part(input)
        spectrum_parameters = self.forward_part(geometric_parameters)
        return geometric_parameters,spectrum_parameters

if __name__ == "__main__":
    print("print number of parameters in the model")
    model = baseline_inverse_forward_model()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
