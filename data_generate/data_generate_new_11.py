import numpy as np
from transform_rbbox import (polygonToRotRectangle_batch, RotBox2Polys_torch, RotBox2Polys, obb2poly)
import torch
from ops.box_iou_rotated import obb_overlaps
import copy

def data_generator(N):
    # polys1 = np.zeros((0, 8))
    # rbboxes1 = np.zeros((0, 5))
    polys11 = np.zeros((0, 8))
    rbboxes11 = np.zeros((0, 5))
    polys12 = np.zeros((0, 8))
    rbboxes12 = np.zeros((0, 5))
    count = 0
    while count < (N//4):
        xy11 = np.random.uniform(low=0, high=1, size=(1, 2))
        wh11 = np.random.uniform(low=0, high=1, size=(1, 2))
        theta11 = np.random.uniform(low=-np.pi, high=np.pi, size=(1, 1))
        rbbox11 = np.concatenate([xy11, wh11, theta11], axis=1)

        xy12 = np.random.uniform(low=0, high=1, size=(1, 2))
        wh12 = np.random.uniform(low=0, high=1, size=(1, 2))
        theta12 = np.random.uniform(low=-np.pi, high=np.pi, size=(1, 1))
        rbbox12 = np.concatenate([xy12, wh12, theta12], axis=1)

        poly11 = RotBox2Polys(rbbox11)
        large_poly11 = poly11 * 500  #放大框，便于obb_overlap的计算
        rbbox11 = polygonToRotRectangle_batch(large_poly11, with_module=False)
        poly12 = RotBox2Polys(rbbox12)
        large_poly12 = poly12 * 500  # 放大框，便于obb_overlap的计算
        rbbox12 = polygonToRotRectangle_batch(large_poly12, with_module=False)

        polys11 = np.concatenate([polys11, large_poly11], axis=0)    #放大后的8参数框
        rbboxes11 = np.concatenate([rbboxes11, rbbox11], axis=0)   #未归一化的5参数框
        polys12 = np.concatenate([polys12, large_poly12], axis=0)  # 放大后的8参数框
        rbboxes12 = np.concatenate([rbboxes12, rbbox12], axis=0)  # 未归一化的5参数框
        count += 1

    IoU_targets = obb_overlaps(rbboxes11, rbboxes12, is_aligned=False)
    match_ind = np.argmax(IoU_targets, axis=1)
    rbboxes12 = rbboxes12[match_ind]
    rbboxes1 = np.concatenate([rbboxes11, rbboxes12], axis=0)
    rbboxes2_match = np.concatenate([rbboxes12, rbboxes11], axis=0)

    polys12 = polys12[match_ind]
    polys1 = np.concatenate([polys11, polys12], axis=0)
    polys2_match = np.concatenate([polys12, polys11], axis=0)


    while (N // 4) <= count < (N // 2):
        xy21 = np.random.uniform(low=0, high=1, size=(1, 2))
        wh21 = np.random.uniform(low=0, high=1, size=(1, 2))
        theta21 = np.random.uniform(low=-np.pi, high=np.pi, size=(1, 1))
        rbbox21 = np.concatenate([xy21, wh21, theta21], axis=1)

        p2 = np.random.rand() * 0.05
        xy22 = xy21 * np.random.uniform(low=1.0-p2, high=1.0+p2, size=(1, 2))
        wh22 = wh21 * np.random.uniform(low=1.0-p2, high=1.0+p2, size=(1, 2))
        theta22 = theta21 * np.random.uniform(low=1.0-p2, high=1.0+p2, size=(1, 1))
        rbbox22 = np.concatenate([xy22, wh22, theta22], axis=1)

        poly21 = RotBox2Polys(rbbox21)
        large_poly21 = poly21 * 500  # 放大框，便于obb_overlap的计算
        rbbox21 = polygonToRotRectangle_batch(large_poly21, with_module=False)

        poly22 = RotBox2Polys(rbbox22)
        large_poly22 = poly22 * 500 # 放大框，便于obb_overlap的计算
        rbbox22 = polygonToRotRectangle_batch(large_poly22, with_module=False)

        polys1 = np.concatenate([polys1, large_poly21, large_poly22], axis=0)  # 归一化后的8参数框
        rbboxes1 = np.concatenate([rbboxes1, rbbox21, rbbox22], axis=0)  # 未归一化的5参数框
        polys2_match = np.concatenate([polys2_match, large_poly22, large_poly21], axis=0)  # 归一化后的8参数框
        rbboxes2_match = np.concatenate([rbboxes2_match, rbbox22, rbbox21], axis=0)  # 未归一化的5参数框
        count += 1

    # 打乱框的顺序
    shuffle_ind = np.arange(N)
    np.random.shuffle(shuffle_ind)
    rbboxes1 = rbboxes1[shuffle_ind]
    rbboxes2_match = rbboxes2_match[shuffle_ind]
    polys1 = polys1[shuffle_ind]
    polys2_match = polys2_match[shuffle_ind]

    polys1 = torch.Tensor(polys1).cuda()
    polys2_match = torch.Tensor(polys2_match).cuda()
    rbboxes1 = torch.Tensor(rbboxes1).cuda()
    rbboxes2_match = torch.Tensor(rbboxes2_match).cuda()

    #归一化
    x = torch.cat([polys1[:, 0::2], polys2_match[:, 0::2]], dim=1)  # N,8
    xmin, _ = torch.min(x, dim=1)
    xmax, _ = torch.max(x, dim=1)
    y = torch.cat([polys1[:, 1::2], polys2_match[:, 1::2]], dim=1)  # N,8
    ymin, _ = torch.min(y, dim=1)  # N
    ymax, _ = torch.max(y, dim=1)
    # enclose_bbox_point1 = torch.cat([xmin.unsqueeze(-1), ymin.unsqueeze(-1)], dim=1).detach().cpu().numpy()
    # enclose_bbox_point1 = enclose_bbox_point1.astype(int)
    # enclose_bbox_point2 = torch.cat([xmax.unsqueeze(-1), ymax.unsqueeze(-1)], dim=-1).detach().cpu().numpy()
    # enclose_bbox_point2 = enclose_bbox_point2.astype(int)
    w = torch.sub(xmax, xmin).unsqueeze(-1)
    h = torch.sub(ymax, ymin).unsqueeze(-1)
    long_side, _ = torch.max(torch.cat([w, h], dim=1), dim=1)  # N

    polys1[:, 0::2] = polys1[:, 0::2] - xmin.unsqueeze(-1)
    polys1[:, 1::2] = polys1[:, 1::2] - ymin.unsqueeze(-1)
    # pos_poly_pred_decode_new = torch.zeros_like(pos_poly_pred_decode)
    polys1 = polys1 / long_side.unsqueeze(-1)

    polys2_match[:, 0::2] = torch.sub(polys2_match[:, 0::2], xmin.unsqueeze(-1))
    polys2_match[:, 1::2] = torch.sub(polys2_match[:, 1::2], ymin.unsqueeze(-1))
    # pos_poly_target_decode_new = torch.zeros_like(pos_poly_target_decode)
    polys2_match = polys2_match / long_side.unsqueeze(-1)

    return polys1, polys2_match, rbboxes1, rbboxes2_match
    #return polys1, polys2_match, rbboxes1, rbboxes2_match, enclose_bbox_point1, enclose_bbox_point2






