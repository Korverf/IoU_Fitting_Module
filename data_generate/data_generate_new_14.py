import numpy as np
from transform_rbbox import (polygonToRotRectangle_batch, RotBox2Polys_torch, RotBox2Polys, obb2poly)
import torch
from ops.box_iou_rotated import obb_overlaps
import copy

#不加随机框生成算法，加框归一化算法。

def data_generator(N):
    polys1 = np.zeros((0, 8))
    rbboxes1 = np.zeros((0, 5))
    polys2 = np.zeros((0, 8))
    rbboxes2 = np.zeros((0, 5))
    count = 0
    while count < (N):
        xy1 = np.random.uniform(low=0, high=1, size=(1, 2))
        wh1 = np.random.uniform(low=0, high=1, size=(1, 2))
        theta1 = np.random.uniform(low=-np.pi, high=np.pi, size=(1, 1))
        rbbox1 = np.concatenate([xy1, wh1, theta1], axis=1)

        xy2 = np.random.uniform(low=0, high=1, size=(1, 2))
        wh2 = np.random.uniform(low=0, high=1, size=(1, 2))
        theta2 = np.random.uniform(low=-np.pi, high=np.pi, size=(1, 1))
        rbbox2 = np.concatenate([xy2, wh2, theta2], axis=1)

        poly1 = RotBox2Polys(rbbox1)
        large_poly1 = poly1 * 500  #放大框，便于obb_overlap的计算
        rbbox1 = polygonToRotRectangle_batch(large_poly1, with_module=False)
        poly2 = RotBox2Polys(rbbox2)
        large_poly2 = poly2 * 500  # 放大框，便于obb_overlap的计算
        rbbox2 = polygonToRotRectangle_batch(large_poly2, with_module=False)

        polys1 = np.concatenate([polys1, poly1], axis=0)    #放大后的8参数框
        rbboxes1 = np.concatenate([rbboxes1, rbbox1], axis=0)   #未归一化的5参数框
        polys2 = np.concatenate([polys2, poly2], axis=0)  # 放大后的8参数框
        rbboxes2 = np.concatenate([rbboxes2, rbbox2], axis=0)  # 未归一化的5参数框
        count += 1

    polys1 = torch.Tensor(polys1).cuda()
    polys2 = torch.Tensor(polys2).cuda()
    rbboxes1 = torch.Tensor(rbboxes1).cuda()
    rbboxes2 = torch.Tensor(rbboxes2).cuda()
    #归一化
    x = torch.cat([polys1[:, 0::2], polys2[:, 0::2]], dim=1)  # N,8
    xmin, _ = torch.min(x, dim=1)
    xmax, _ = torch.max(x, dim=1)
    y = torch.cat([polys1[:, 1::2], polys2[:, 1::2]], dim=1)  # N,8
    ymin, _ = torch.min(y, dim=1)  # N
    ymax, _ = torch.max(y, dim=1)

    w = torch.sub(xmax, xmin).unsqueeze(-1)
    h = torch.sub(ymax, ymin).unsqueeze(-1)
    long_side, _ = torch.max(torch.cat([w, h], dim=1), dim=1)  # N

    polys1[:, 0::2] = polys1[:, 0::2] - xmin.unsqueeze(-1)
    polys1[:, 1::2] = polys1[:, 1::2] - ymin.unsqueeze(-1)
    polys1 = polys1 / long_side.unsqueeze(-1)

    polys2[:, 0::2] = torch.sub(polys2[:, 0::2], xmin.unsqueeze(-1))
    polys2[:, 1::2] = torch.sub(polys2[:, 1::2], ymin.unsqueeze(-1))
    polys2 = polys2 / long_side.unsqueeze(-1)

    return polys1, polys2, rbboxes1, rbboxes2






