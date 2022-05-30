import torch.nn as nn
# from mmcv.cnn import (Linear, build_activation_layer, build_norm_layer,
#                       xavier_init)
import torch
#from torch.nn.modules import loss
# from torch.nn.modules.linear import _LinearWithBias
# from torch.nn.init import xavier_uniform_
# from torch.nn.init import constant_
# from torch.nn.init import xavier_normal_
# from torch.nn import functional as F
# import numpy as np
from box_iou_rotated import obb_overlaps
#from transform_rbbox import (polygonToRotRectangle_batch, RotBox2Polys_torch, RotBox2Polys, obb2poly)


class IOUfitModule(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super(IOUfitModule, self).__init__()
        self.conv1 = Conv(in_channels=in_features, out_channels=512, kernel_size=1)
        self.bottle1 = Bottleneck(in_channels=512)
        self.conv2 = Conv(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2) #256
        self.bottle2 = Bottleneck(in_channels=256)
        self.conv3 = Conv(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2) #128
        self.bottle3 = Bottleneck(in_channels=128)
        self.conv4 = Conv(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2) #64
        #self.bottle4 = Bottleneck(in_channels=64)
        self.conv5 = Conv(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2) #32
        self.conv6 = Conv(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2) #16
        self.conv7 = Conv(in_channels=16, out_channels=1, kernel_size=1, with_act=False)

        # self.out_fc = Linear(hidden_features,1)
        #self.sigmoid = nn.Sigmoid()
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.loss_weight = 5.0
        #self.scale_factor = 1

    def forward(self, pred, gt):
        input = torch.cat([pred, gt], dim=-1) #N,16
        input = torch.unsqueeze(input, dim=-1)  # N,16,1
        out = self.conv1(input) #N,512,1
        out = out.transpose(1,2) # N,1,512
        out = self.bottle1(out) # N,1,512
        out = self.conv2(out) #N,1,256
        out = self.bottle2(out)  # N,1,256
        out = self.conv3(out) #N,1,128
        out = self.bottle3(out)  # N,1,128
        out = self.conv4(out) #N,1,64
        #out = self.bottle4(out)  # N,1,64
        out = self.conv5(out) #N,1,32
        out = self.conv6(out) #N,1,16
        out = out.transpose(1,2)  #N,16,1
        out = self.conv7(out) #N,1,1
        out = torch.squeeze(out, dim=-1)
        return out

    def loss(self, rbboxes1, rbboxes2, iou_fit_value):
        IoU_targets = obb_overlaps(rbboxes1, rbboxes2.detach(), is_aligned=True).squeeze(
                    1).clamp(min=1e-6, max=1)

        loss_fit = self.mse_loss(iou_fit_value, IoU_targets).sqrt()
        loss_fit = self.loss_weight * loss_fit
        #loss_fit = ((iou_fit_value - IoU_targets.detach()).square() + 1).log().sqrt()

        return loss_fit

class Bottleneck(nn.Module):
    def __init__(self, in_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        #self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=in_channels // 4, out_channels=in_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, input):#  N,1,C
        input = input.transpose(1, 2) #  N,C,1
        out = self.relu(self.conv1(input))#N,C/4,1
        # out = out.transpose(1, 2) # N,1,C/4
        # out = self.relu(self.conv2(out))
        # out = out.transpose(1, 2)  #N,C/4,1
        #out = self.relu(input + self.conv3(out))#  N,C,1
        out = self.relu(self.conv3(out))  # N,C,1
        out = out.transpose(1, 2)  # N,1,C
        return out

class Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 with_act=True,
                 with_bn=False,
                 act_func='ReLU',
                 add_residual=False):
        super(Conv, self).__init__()
        # self.in_channels = in_channels
        # self.feedforward_channels = feedforward_channels
        # self.act_cfg = act_cfg
        self.add_residual = add_residual
        layers = nn.ModuleList()
        #self.relu = build_activation_layer(dict(type='ReLU'))
        layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding))
        if with_bn == True:
            layers.append(nn.BatchNorm1d(out_channels))
        if with_act == True:
            if act_func=='ReLU':
                self.activate = nn.ReLU()
                layers.append(self.activate)

        self.layers = nn.Sequential(*layers)

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + out

class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 feedforward_channels,
                 out_channels,
                 with_act=True,
                 with_bn=True,
                 act_func='ReLU',
                 add_residual=True):
        super(FC, self).__init__()
        # self.in_channels = in_channels
        # self.feedforward_channels = feedforward_channels
        # self.act_cfg = act_cfg
        self.add_residual = add_residual
        layers = nn.ModuleList()
        #self.relu = build_activation_layer(dict(type='ReLU'))
        layers.append(nn.Linear(in_channels, out_channels))
        if with_bn == True:
            layers.append(nn.BatchNorm1d(out_channels))
        if with_act == True:
            if act_func=='ReLU':
                self.activate = nn.ReLU()
                layers.append(self.activate)

        self.layers = nn.Sequential(*layers)

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + out


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        #self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        #x = self.drop(x)
        return x

def get_model(in_features, hidden_features=None):

    model = IOUfitModule(in_features, hidden_features)  # 101
    return model