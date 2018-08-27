import numpy as np
import torch
import torch.nn as nn
import time

class Modified_SmoothL1Loss(torch.nn.Module):

    def __init__(self):
        super(Modified_SmoothL1Loss,self).__init__()

    def forward(self,x,y):
        total_loss = 0
        assert(x.shape == y.shape)
        z = (x - y).float()
        mse = (torch.abs(z) < 0.01).float() * z
        l1 = (torch.abs(z) >= 0.01).float() * z
        total_loss += torch.sum(self._calculate_MSE(mse))
        total_loss += torch.sum(self._calculate_L1(l1))

        return total_loss/z.shape[0]

    def _calculate_MSE(self, z):
        return 0.5 *(torch.pow(z,2))

    def _calculate_L1(self,z):
        return 0.01 * (torch.abs(z) - 0.005)
