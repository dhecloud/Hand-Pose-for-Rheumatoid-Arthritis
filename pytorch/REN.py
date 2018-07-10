from torch import nn
import torch
import numpy as np

class Residual(nn.Module):

    def __init__(self, planes):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= planes, out_channels=planes, kernel_size = 3,  padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.relu(out)

        out += residual
        return out

class REN(nn.Module):

    def __init__(self, num_joints=63):
        super(REN, self).__init__()
        #nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size = 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size = 3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2_dim_inc = nn.Conv2d(in_channels=16, out_channels=32, kernel_size = 1, padding=0)
        self.res1 = Residual(planes = 32)
        self.conv3_dim_inc = nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 1, padding=0)
        self.res2 = Residual(planes = 64)
        self.fc1 = nn.Linear(64, 63)

    def forward(self, x):

        out = self.conv0(x)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.maxpool(out)
        out = self.relu(out)

        out = self.conv2_dim_inc(out)
        out = self.relu(out)

        out = self.res1(out)

        out = self.maxpool(out)
        out = self.relu(out)

        out = self.conv3_dim_inc(out)
        out = self.relu(out)

        out = self.res2(out)

        out = self.maxpool(out)
        out = self.relu(out)

        out = self.fc1(out)


        return out

model = REN()
print(model)
depth = np.zeros((1, 96, 96), dtype=np.float32)
print(depth.shape)
output = model(torch.from_numpy(depth))
print(output)
