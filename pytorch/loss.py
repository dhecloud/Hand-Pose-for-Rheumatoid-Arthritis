import numpy as np
import torch
import torch.nn as nn

class Modified_SmoothL1Loss(torch.nn.Module):

    def __init__(self):
        super(Modified_SmoothL1Loss,self).__init__()

    def forward(self,x,y):
        total_loss = 0
        z = x - y
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                total_loss += self._smooth_l1(z[i][j])

        return total_loss/z.shape[0]

    def _smooth_l1(self, z):
        if torch.abs(z) < 0.01:
            loss = self._calculate_MSE(z)
        else:
            loss = self._calculate_L1(z)

        return loss

    def _calculate_MSE(self, z):
        return 0.5 *(torch.pow(z,2))

    def _calculate_L1(self,z):
        return 0.01 * (torch.abs(z) - 0.005)
