from torch import nn
import torch
import numpy as np

class RegionEnsemble(nn.Module):

    def __init__(self):
        super(RegionEnsemble, self).__init__()
        self.grids = nn.ModuleList()
        for i in range(36):
            self.grids.append(self.make_block())

    def make_block(self):
        return nn.Sequential(nn.Linear(64*2*2, 2048), nn.ReLU(), nn.Dropout(), nn.Linear(2048,2048), nn.ReLU(), nn.Dropout())

    def forward(self, x):
        # half1 = x[:, :, :6,:]
        # half2 = x[:, :, 6:,:]
        #
        # quarter1 = half1[:, :, :, :6]   #64x6x6
        # quarter2 = half1[:, :, :, 6:]
        # quarter3 = half2[:, :, :, :6]
        # quarter4 = half2[:, :, :, 6:]

        ensemble = []
        k = 0
        for i in range(0,12,2):
            for j in range(0,12,2):
                out = x[:, :, i:i+2, j:j+2]
                out = out.contiguous()
                out = out.view(out.size(0),-1)
                out = self.grids[k](out)
                k +=1
                ensemble.append(out)

        #
        # quarter1 = quarter1.contiguous()
        # quarter1 = quarter1.view(quarter1.size(0),-1)
        # quarter1 = self.fc1(quarter1)
        # quarter1 = self.relu(quarter1)
        # quarter1 = self.dropout(quarter1)
        # quarter1 = self.fc2(quarter1)
        # quarter1 = self.relu(quarter1)
        # quarter1 = self.dropout(quarter1)
        #
        # quarter2 = quarter2.contiguous()
        # quarter2 = quarter2.view(quarter2.size(0),-1)
        # quarter2 = self.fc1(quarter2)
        # quarter2 = self.relu(quarter2)
        # quarter2 = self.dropout(quarter2)
        # quarter2 = self.fc2(quarter2)
        # quarter2 = self.relu(quarter2)
        # quarter2 = self.dropout(quarter2)
        #
        # quarter3 = quarter3.contiguous()
        # quarter3 = quarter3.view(quarter3.size(0),-1)
        # quarter3 = self.fc1(quarter3)
        # quarter3 = self.relu(quarter3)
        # quarter3 = self.dropout(quarter3)
        # quarter3 = self.fc2(quarter3)
        # quarter3 = self.relu(quarter3)
        # quarter3 = self.dropout(quarter3)
        #
        # quarter4 = quarter4.contiguous()
        # quarter4 = quarter4.view(quarter4.size(0),-1)
        # quarter4 = self.fc1(quarter4)
        # quarter4 = self.relu(quarter4)
        # quarter4 = self.dropout(quarter4)
        # quarter4 = self.fc2(quarter4)
        # quarter4 = self.relu(quarter4)
        # quarter4 = self.dropout(quarter4)


        out = torch.cat(ensemble,1)

        return out



class Residual(nn.Module):

    def __init__(self, planes):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= planes, out_channels=planes, kernel_size = 3,  padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv1(out)

        out += residual
        return out

class REN(nn.Module):

    def __init__(self, num_joints=63):
        super(REN, self).__init__()
        #nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size = 3, padding=1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size = 3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2_dim_inc = nn.Conv2d(in_channels=16, out_channels=32, kernel_size = 1, padding=0)
        self.res1 = Residual(planes = 32)
        self.conv3_dim_inc = nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 1, padding=0)
        self.res2 = Residual(planes = 64)
        self.dropout = nn.Dropout()
        self.region_ens = RegionEnsemble()
        #class torch.nn.Linear(in_features, out_features, bias=True)
        self.fc1 = nn.Linear(36*2048, 63)

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
        out = self.relu(out)        #relu5
        out = self.dropout(out)


        #slice
        out = self.region_ens(out)
        # flatten the output
        out = out.view(out.size(0),-1)

        out = self.fc1(out)
        return out



# test = torch.tensor(np.random.rand(1,1,96,96))
# test = test.cuda()
# test = test.double()
# model = REN()
# model = model.double()
# model = model.cuda()
# print(model)
# result = model(test)
# print(result)
