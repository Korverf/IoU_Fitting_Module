import torch.nn as nn
from mmcv.cnn import (Linear, build_activation_layer, build_norm_layer,
                      xavier_init)
import torch
from torch.nn.modules.linear import _LinearWithBias
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn import functional as F
import numpy as np
from box_iou_rotated import obb_overlaps


class IOUfitModule(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super(IOUfitModule, self).__init__()
        # self.fc1 = FC(in_channels=in_features, feedforward_channels=hidden_features, out_channels=hidden_features, 
        #               act_cfg=dict(type='GELU'), add_residual=False)
        # self.fc2 = FC(in_channels=in_features, feedforward_channels=hidden_features, out_channels=hidden_features, 
        #               act_cfg=dict(type='GELU'), add_residual=False)
        # self.fc3 = FC(in_channels=in_features, feedforward_channels=hidden_features, out_channels=hidden_features,
        #               act_cfg=dict(type='ReLU'), add_residual=False)
        # self.fc4 = FC(in_channels=in_features, feedforward_channels=hidden_features, out_channels=hidden_features,
        #               act_cfg=dict(type='ReLU'), add_residual=False)
        self.fc5 = FC(in_channels=in_features, feedforward_channels=2048, out_channels=1,
                      act_cfg=dict(type='Sigmoid'), add_residual=False, with_act=False, with_bn=False)
        # self.fc3 = FC(in_channels=in_features, feedforward_channels=hidden_features, out_channels=1,
        #               with_act=False)

        # self.out_fc = Linear(hidden_features,1)
        #self.sigmoid = nn.Sigmoid()
        self.mse_loss = nn.MSELoss(reduction='sum')
        #self.scale_factor = 1

    def forward(self, pred, gt):
        input = torch.cat([pred, gt], dim=-1)
        # out = self.fc1(input)
        # out = self.fc2(out)
        # out = self.fc3(out)
        # out = self.fc4(out)
        #out = out * self.scale_factor
        # out = self.fc5(out)
        #out = self.sigmoid(out)
        out = self.fc5(input)
        return out

    def loss(self, pred_decode, gt_decode, iou_fit_value):
        IoU_targets = obb_overlaps(pred_decode, gt_decode.detach(), is_aligned=True).squeeze(
                    1).clamp(min=1e-6)

        loss_fit = self.mse_loss(iou_fit_value, IoU_targets.detach())#.sqrt()
        #loss_fit = ((iou_fit_value - IoU_targets.detach()).square() + 1).log().sqrt()

        return loss_fit


class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 feedforward_channels,
                 out_channels,
                 with_act=True,
                 with_bn=True,
                 act_cfg=dict(type='ReLU'),
                 add_residual=True):
        super(FC, self).__init__()
        # self.in_channels = in_channels
        # self.feedforward_channels = feedforward_channels
        # self.act_cfg = act_cfg
        self.add_residual = add_residual
        layers = nn.ModuleList()
        self.relu = build_activation_layer(dict(type='ReLU'))
        layers.append(nn.Sequential(Linear(in_channels, feedforward_channels),
                                    self.relu, 
                                    Linear(feedforward_channels, out_channels)))
        if with_bn == True:
            layers.append(nn.BatchNorm1d(out_channels))
        if with_act == True:
            self.activate2 = build_activation_layer(act_cfg)
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