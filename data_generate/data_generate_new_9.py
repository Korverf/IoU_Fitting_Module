import numpy as np
from transform_rbbox import (polygonToRotRectangle_batch, RotBox2Polys_torch, RotBox2Polys, obb2poly)
import torch
from box_iou_rotated import obb_overlaps
import copy

def data_generator(N):
    #N = np.random.randint(low=1, high=10, size=None, dtype=int)
    # polys1 = np.zeros((0, 8))
    # rbboxes1 = np.zeros((0, 5))
    # polys2 = np.zeros((0, 8))
    # rbboxes2 = np.zeros((0, 5))

    polys11 = np.zeros((0, 8))
    rbboxes11 = np.zeros((0, 5))
    polys12 = np.zeros((0, 8))
    rbboxes12 = np.zeros((0, 5))

    polys21 = np.zeros((0, 8))
    rbboxes21 = np.zeros((0, 5))
    polys22 = np.zeros((0, 8))
    rbboxes22 = np.zeros((0, 5))

    polys31 = np.zeros((0, 8))
    rbboxes31 = np.zeros((0, 5))
    polys32 = np.zeros((0, 8))
    rbboxes32 = np.zeros((0, 5))
    count = 0
    while count < N:
        p = np.random.rand() * 3
        if p <= 1:  #随机生成框，然后匹配最大IOU
            xy11 = np.random.uniform(low=-1, high=1, size=(1, 2))
            wh11 = np.random.uniform(low=0.01, high=1.6, size=(1, 2))
            theta11 = np.random.uniform(low=-np.pi, high=np.pi, size=(1, 1))
            rbbox11 = np.concatenate([xy11, wh11, theta11], axis=1)

            xy12 = np.random.uniform(low=-1, high=1, size=(1, 2))
            wh12 = np.random.uniform(low=0.01, high=1.6, size=(1, 2))
            theta12 = np.random.uniform(low=-np.pi, high=np.pi, size=(1, 1))
            rbbox12 = np.concatenate([xy12, wh12, theta12], axis=1)

            poly11 = RotBox2Polys(rbbox11)
            poly12 = RotBox2Polys(rbbox12)
            large_poly11 = poly11 * 400 + 400  #放大框，便于obb_overlap的计算
            large_poly12 = poly12 * 400 + 400
            rbbox11 = polygonToRotRectangle_batch(large_poly11, with_module=False)
            rbbox12 = polygonToRotRectangle_batch(large_poly12, with_module=False)

            xmin11, xmax11, ymin11, ymax11 = np.min(poly11[:,0::2]), np.max(poly11[:,0::2]), np.min(poly11[:,1::2]), np.max(poly11[:,1::2])
            xmin12, xmax12, ymin12, ymax12 = np.min(poly12[:, 0::2]), np.max(poly12[:, 0::2]), np.min(poly12[:, 1::2]), np.max(
                poly12[:, 1::2])
            if (xmin11 > -1.2) & (xmax11 < 1.2) & (ymin11 > -1.2) & (ymax11 < 1.2) & (xmin12 > -1.2) & (xmax12 < 1.2) & (ymin12 > -1.2) & (ymax12 < 1.2):
                polys11 = np.concatenate([polys11, poly11], axis=0)    #归一化后的8参数框
                rbboxes11 = np.concatenate([rbboxes11, rbbox11], axis=0)   #未归一化的5参数框
                polys12 = np.concatenate([polys12, poly12], axis=0)  # 归一化后的8参数框
                rbboxes12 = np.concatenate([rbboxes12, rbbox12], axis=0)  # 未归一化的5参数框
                count += 1
        elif 1 < p <= 2:  #随机生成框1，在框1的周围生成极高重叠度(IOU>0.9)的框2.
            xy21 = np.random.uniform(low=-1, high=1, size=(1, 2))
            wh21 = np.random.uniform(low=0.01, high=1.6, size=(1, 2))
            theta21 = np.random.uniform(low=-np.pi, high=np.pi, size=(1, 1))
            rbbox21 = np.concatenate([xy21, wh21, theta21], axis=1)

            xy22 = xy21 * np.random.uniform(low=0.9, high=1.1, size=(1, 2))
            xy22 = np.min([xy22, np.ones(shape=(1,2))], axis=0)
            xy22 = np.max([xy22, -1 * np.ones(shape=(1, 2))], axis=0)
            wh22 = wh21 * np.random.uniform(low=0.9, high=1.1, size=(1, 2))
            theta22 = theta21 * np.random.uniform(low=0.9, high=1.1, size=(1, 1))
            theta22 = np.min([theta22, np.pi * np.ones(shape=(1,1))], axis=0)
            theta22 = np.max([theta22, -np.pi * np.ones(shape=(1, 1))], axis=0)
            rbbox22 = np.concatenate([xy22, wh22, theta22], axis=1)

            poly21 = RotBox2Polys(rbbox21)
            large_poly21 = poly21 * 400 + 400  # 放大框，便于obb_overlap的计算
            rbbox21 = polygonToRotRectangle_batch(large_poly21, with_module=False)

            poly22 = RotBox2Polys(rbbox22)
            large_poly22 = poly22 * 400 + 400  # 放大框，便于obb_overlap的计算
            rbbox22 = polygonToRotRectangle_batch(large_poly22, with_module=False)

            xmin21, xmax21, ymin21, ymax21 = np.min(poly21[:, 0::2]), np.max(poly21[:, 0::2]), np.min(poly21[:, 1::2]), np.max(
                poly21[:, 1::2])
            xmin22, xmax22, ymin22, ymax22 = np.min(poly22[:, 0::2]), np.max(poly22[:, 0::2]), np.min(poly22[:, 1::2]), np.max(
                poly22[:, 1::2])
            if (xmin21 > -1.2) & (xmax21 < 1.2) & (ymin21 > -1.2) & (ymax21 < 1.2) & (xmin22 > -1.2) & (xmax22 < 1.2) \
                    & (ymin22 > -1.2) & (ymax22 < 1.2):
                polys21 = np.concatenate([polys21, poly21], axis=0)  # 归一化后的8参数框
                rbboxes21 = np.concatenate([rbboxes21, rbbox21], axis=0)  # 未归一化的5参数框
                polys22 = np.concatenate([polys22, poly22], axis=0)  # 归一化后的8参数框
                rbboxes22 = np.concatenate([rbboxes22, rbbox22], axis=0)  # 未归一化的5参数框
                count += 1
        else:  #随机生成尺度很小的框1，在框1的周围生成较高重叠度(IOU>0.6)的框2.
            xy31 = np.random.uniform(low=-1, high=1, size=(1, 2))
            wh31 = np.random.uniform(low=0.01, high=0.1, size=(1, 2))
            theta31 = np.random.uniform(low=-np.pi, high=np.pi, size=(1, 1))
            rbbox31 = np.concatenate([xy31, wh31, theta31], axis=1)

            xy32 = xy31 * np.random.uniform(low=0.8, high=1.2, size=(1, 2))
            xy32 = np.min([xy32, np.ones(shape=(1,2))], axis=0)
            xy32 = np.max([xy32, -1 * np.ones(shape=(1, 2))], axis=0)
            wh32 = wh31 * np.random.uniform(low=0.8, high=1.2, size=(1, 2))
            theta32 = theta31 * np.random.uniform(low=0.8, high=1.2, size=(1, 1))
            theta32 = np.min([theta32, np.pi * np.ones(shape=(1,1))], axis=0)
            theta32 = np.max([theta32, -np.pi * np.ones(shape=(1, 1))], axis=0)
            rbbox32 = np.concatenate([xy32, wh32, theta32], axis=1)

            poly31 = RotBox2Polys(rbbox31)
            large_poly31 = poly31 * 400 + 400  # 放大框，便于obb_overlap的计算
            rbbox31 = polygonToRotRectangle_batch(large_poly31, with_module=False)

            poly32 = RotBox2Polys(rbbox32)
            large_poly32 = poly32 * 400 + 400  # 放大框，便于obb_overlap的计算
            rbbox32 = polygonToRotRectangle_batch(large_poly32, with_module=False)

            xmin31, xmax31, ymin31, ymax31 = np.min(poly31[:, 0::2]), np.max(poly31[:, 0::2]), np.min(poly31[:, 1::2]), np.max(
                poly31[:, 1::2])
            xmin32, xmax32, ymin32, ymax32 = np.min(poly32[:, 0::2]), np.max(poly32[:, 0::2]), np.min(poly32[:, 1::2]), np.max(
                poly32[:, 1::2])
            if (xmin31 > -1.2) & (xmax31 < 1.2) & (ymin31 > -1.2) & (ymax31 < 1.2) & (xmin32 > -1.2) & (xmax32 < 1.2) \
                    & (ymin32 > -1.2) & (ymax32 < 1.2):
                polys31 = np.concatenate([polys31, poly31], axis=0)  # 归一化后的8参数框
                rbboxes31 = np.concatenate([rbboxes31, rbbox31], axis=0)  # 未归一化的5参数框
                polys32 = np.concatenate([polys32, poly32], axis=0)  # 归一化后的8参数框
                rbboxes32 = np.concatenate([rbboxes32, rbbox32], axis=0)  # 未归一化的5参数框
                count += 1

    IoU_targets = obb_overlaps(rbboxes11, rbboxes12, is_aligned=False)
    match_ind = np.argmax(IoU_targets, axis=1)
    rbboxes12 = copy.deepcopy(rbboxes11)
    polys12 = copy.deepcopy(polys11)
    rbboxes12_match = rbboxes12[match_ind]
    polys12_match = polys12[match_ind]

    rbboxes1 = np.concatenate([rbboxes11, rbboxes21, rbboxes31], axis=0)
    rbboxes2 = np.concatenate([rbboxes12_match, rbboxes22, rbboxes32], axis=0)
    polys1 = np.concatenate([polys11, polys21, polys31], axis=0)
    polys2 = np.concatenate([polys12_match, polys22, polys32], axis=0)

    #打乱框的顺序
    shuffle_ind = np.arange(N)
    np.random.shuffle(shuffle_ind)
    rbboxes1 = rbboxes1[shuffle_ind]
    rbboxes2 = rbboxes2[shuffle_ind]
    polys1 = polys1[shuffle_ind]
    polys2 = polys2[shuffle_ind]

    #IoU_targets2 = obb_overlaps(rbboxes1, rbboxes2_match, is_aligned=True)
    polys1 = torch.Tensor(polys1).cuda()
    polys2 = torch.Tensor(polys2).cuda()
    rbboxes1 = torch.Tensor(rbboxes1).cuda()
    rbboxes2 = torch.Tensor(rbboxes2).cuda()

    return polys1, polys2, rbboxes1, rbboxes2






