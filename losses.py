import torch.nn as nn
import torch
import numpy as np
class multiloss(nn.Module):
    def __init__(self, objective_num,losses_fns):
        super(multiloss, self).__init__()
        self.objective_num = objective_num
        self.log_var = nn.Parameter(torch.zeros(self.objective_num))
        self.losses_fns = losses_fns

    def forward(self, output, target):
        loss = 0
        for i in range(len(self.losses_fns)):
            loss_fn = self.losses_fns[i]
            precision = torch.exp(-self.log_var[i])
            loss += precision * loss_fn(output, target) + self.log_var[i]
        return loss

class CircularLoss(nn.Module):
    def forward(self, y_true, y_pred):
        # Calculate the circular mean of predicted and true angles
        delta_theta = y_pred - y_true
        cos_delta_theta = torch.cos(delta_theta)
        loss = 1 - cos_delta_theta
        return loss.mean()
class gamma_continuity_loss(nn.Module):
    def __init__(self,lamda):
        super(gamma_continuity_loss,self).__init__()
        self.lamda = lamda # regularization parameter
    def forward(self,gamma,target=None):
        gamma = torch.tensor(gamma)
        gamma_magnitude = gamma[:,:gamma.shape[1]//2]
        first_der = torch.diff(gamma_magnitude,dim=1)
        second_der_sqrd = torch.square(torch.diff(first_der,dim=1))
        smoothness_loss = self.lamda*torch.mean(second_der_sqrd)
        return smoothness_loss

class CustomHuberLoss(nn.Module):
    def __init__(self, delta=1.0,mag_continuity_lamda=0):
        super().__init__()
        self.delta = delta
        self.mag_continuity_lamda = mag_continuity_lamda
    def forward(self, pred, target):
        mag_con_loss_fn = gamma_continuity_loss(self.mag_continuity_lamda)
        mag_con_loss = mag_con_loss_fn(pred)
        diff = torch.abs(pred - target)
        huber_loss = torch.where(diff < self.delta, 0.5 * diff**2, self.delta * (diff-0.5*self.delta)).mean()
        return huber_loss+mag_con_loss

class gamma_loss(nn.Module):
    def __init__(self,delta=1.0):
        super(gamma_loss,self).__init__()
        self.delta = delta
    def forward(self,gamma,target):
        gamma_magnitude = gamma[:,:gamma.shape[1]//2]
        gamma_phase = gamma[:,gamma.shape[1]//2:]
        target_magnitude = target[:,:target.shape[1]//2]
        target_phase = target[:,target.shape[1]//2:]
        gamma_x,gamma_y = gamma_magnitude*torch.cos(gamma_phase),gamma_magnitude*torch.sin(gamma_phase)
        target_x,target_y = target_magnitude*torch.cos(target_phase),target_magnitude*torch.sin(target_phase)
        diff = torch.abs(gamma_x - target_x) + torch.abs(gamma_y - target_y)
        loss = torch.where(diff < self.delta, 0.5 * ( torch.square(gamma_x - target_x) + torch.square(gamma_y - target_y) ), self.delta * (diff-0.5*self.delta)).mean()
        return loss

class gamma_loss_dB(nn.Module):
    def __init__(self):
        super(gamma_loss_dB,self).__init__()
        self.phase_loss = CircularLoss()
        self.dB_magnitude_loss = nn.HuberLoss()
    def forward(self,pred,target):
        pred_magnitude = pred[:,:pred.shape[1]//2]
        pred_phase = pred[:,pred.shape[1]//2:]
        target_magnitude = target[:,:target.shape[1]//2]
        target_phase = target[:,target.shape[1]//2:]
        loss = self.dB_magnitude_loss(pred_magnitude,target_magnitude) + self.phase_loss(pred_phase,target_phase)
        return loss


