from torch import nn
import torch
import numpy as np

class RegionEnsemble(nn.Module):

    def __init__(self, feat_size=12):
        assert((feat_size/4).is_integer())
        super(RegionEnsemble, self).__init__()
        self.feat_size = feat_size
        self.grids = nn.ModuleList()
        for i in range(9):
            self.grids.append(self.make_block(self.feat_size))

    def make_block(self, feat_size):
        size = int(self.feat_size/2)
        return nn.Sequential(nn.Linear(64*size*size, 2048), nn.ReLU(), nn.Dropout(), nn.Linear(2048,2048), nn.ReLU(), nn.Dropout())

    def forward(self, x):

        midpoint = int(self.feat_size/2)
        quarterpoint1 = int(midpoint/2)
        quarterpoint2 = int(quarterpoint1 + midpoint)
        regions = []
        ensemble = []

        #4 corners
        regions += [x[:, :, :midpoint, :midpoint].clone(), x[:, :, :midpoint, midpoint:].clone(), x[:, :, midpoint:, :midpoint].clone(), x[:, :, midpoint:, midpoint:].clone()]
        # 4 overlapping centers

        regions += [x[:, :, quarterpoint1:quarterpoint2, :midpoint].clone(), x[:, :, quarterpoint1:quarterpoint2, midpoint:].clone(), x[:, :, :midpoint, quarterpoint1:quarterpoint2].clone(), x[:, :, midpoint:, quarterpoint1:quarterpoint2].clone()]
        # middle center
        regions += [x[:, :, quarterpoint1:quarterpoint2, quarterpoint1:quarterpoint2].clone()]

        for i in range(0,9):
            out = regions[i]
            # print(out.shape)
            out = out.contiguous()
            out = out.view(out.size(0),-1)
            out = self.grids[i](out)
            ensemble.append(out)

        out = torch.cat(ensemble,1)

        return out



class Residual(nn.Module):

    def __init__(self, planes):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= planes, out_channels=planes, kernel_size = 3,  padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels= planes, out_channels=planes, kernel_size = 3,  padding=1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out += residual
        return out

class REN(nn.Module):

    def __init__(self, args):
        super(REN, self).__init__()
        feat = np.floor(((args.input_size - 1 -1)/2) +1)
        feat = np.floor(((feat - 1-1)/2) +1)
        feat = np.floor(((feat - 1-1)/2) +1)
        #nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.bn = nn.InstanceNorm2d(1)
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size = 3, padding=1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size = 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2_dim_inc = nn.Conv2d(in_channels=16, out_channels=32, kernel_size = 1, padding=0)
        self.relu2 = nn.ReLU()
        self.res1 = Residual(planes = 32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu3 = nn.ReLU()
        self.conv3_dim_inc = nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 1, padding=0)
        self.relu4 = nn.ReLU()
        self.res2 = Residual(planes = 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.region_ens = RegionEnsemble(feat_size=feat)
        #class torch.nn.Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(9*2048, args.num_joints)

    def forward(self, x):

        out = self.bn(x)
        out = self.conv0(out)
        out = self.relu0(out)

        out = self.conv1(out)
        out = self.maxpool1(out)
        out = self.relu1(out)

        out = self.conv2_dim_inc(out)
        out = self.relu2(out)

        out = self.res1(out)

        out = self.maxpool2(out)
        out = self.relu3(out)

        out = self.conv3_dim_inc(out)
        out = self.relu4(out)

        out = self.res2(out)

        out = self.maxpool3(out)
        out = self.relu5(out)        #relu5
        out = self.dropout(out)


        #slice
        out = self.region_ens(out)
        # flatten the output
        out = out.view(out.size(0),-1)

        out = self.fc1(out)
        return out
