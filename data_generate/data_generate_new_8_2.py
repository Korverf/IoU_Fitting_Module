import numpy as np
from transform_rbbox import (polygonToRotRectangle_batch, RotBox2Polys_torch, RotBox2Polys, obb2poly)
import torch
from box_iou_rotated import obb_overlaps
import copy

def data_generator(N):
    #N = np.random.randint(low=1, high=10, size=None, dtype=int)
    polys1 = np.zeros((0, 8))
    rbboxes1 = np.zeros((0, 5))
    count = 0
    while count < N:
        xy1 = np.random.uniform(low=-1, high=1, size=(1, 2))
        wh1 = np.random.uniform(low=0.01, high=1.6, size=(1, 2))
        theta1 = np.random.uniform(low=-np.pi, high=np.pi, size=(1, 1))
        rbbox1 = np.concatenate([xy1, wh1, theta1], axis=1)

        poly1 = RotBox2Polys(rbbox1)
        large_poly1 = poly1 * 400 + 400  #放大框，便于obb_overlap的计算
        rbbox1 = polygonToRotRectangle_batch(large_poly1, with_module=False)

        xmin1, xmax1, ymin1, ymax1 = np.min(poly1[:,0::2]), np.max(poly1[:,0::2]), np.min(poly1[:,1::2]), np.max(poly1[:,1::2])
        if (xmin1 > -1.2) & (xmax1 < 1.2) & (ymin1 > -1.2) & (ymax1 < 1.2):
            polys1 = np.concatenate([polys1, poly1], axis=0)    #归一化后的8参数框
            rbboxes1 = np.concatenate([rbboxes1, rbbox1], axis=0)   #未归一化的5参数框
            count += 1
    IoU_targets = obb_overlaps(rbboxes1, rbboxes1, is_aligned=False)#.squeeze(
        #1).clamp(min=1e-6)
    #IoU_targets1 = obb_overlaps(rbboxes1, rbboxes2, is_aligned=True)
    i = np.arange(N)
    IoU_targets[i, i] = 0.
    match_ind = np.argmax(IoU_targets, axis=1)
    rbboxes2 = copy.deepcopy(rbboxes1)
    polys2 = copy.deepcopy(polys1)
    rbboxes2_match = rbboxes2[match_ind]
    polys2_match = polys2[match_ind]
    #IoU_targets2 = obb_overlaps(rbboxes1, rbboxes2_match, is_aligned=True)
    polys1 = torch.Tensor(polys1).cuda()
    polys2_match = torch.Tensor(polys2_match).cuda()
    rbboxes1 = torch.Tensor(rbboxes1).cuda()
    rbboxes2_match = torch.Tensor(rbboxes2_match).cuda()

    # 归一化
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






