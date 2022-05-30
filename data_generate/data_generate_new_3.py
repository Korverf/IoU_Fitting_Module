import numpy as np
from transform_rbbox import (polygonToRotRectangle_batch, RotBox2Polys_torch, RotBox2Polys, obb2poly)
import torch

def data_generator(N):
    #N = np.random.randint(low=1, high=10, size=None, dtype=int)
    polys1 = np.zeros((0, 8))
    polys2 = np.zeros((0, 8))
    rbboxes1 = np.zeros((0, 5))
    rbboxes2 = np.zeros((0, 5))
    count = 0
    while count < N:
        xy1 = 2 * np.random.rand(1, 2) - 1  #(-1,1)
        wh1 = 2 * np.random.rand(1, 2)
        theta1 = np.random.uniform(low=-np.pi, high=np.pi, size=(1, 1))
        rbbox1 = np.concatenate([xy1, wh1, theta1], axis=1)

        #diff_factor = np.random.uniform(low=0.1, high=0.3) #IOU almost zero
        #diff_factor = np.random.uniform(low=0.5, high=2)   #IOU 0.4～0.7
        #diff_factor = np.random.uniform(low=10, high=30)    #IOU 0.9～1
        # p = np.random.rand() * 5
        # if 0 <= p < 2:
        #     diff_factor = np.random.uniform(low=0.1, high=0.3)
        # elif 2 <= p < 3:
        #     diff_factor = np.random.uniform(low=0.5, high=2)
        # elif 3 <= p < 4:
        #     diff_factor = np.random.uniform(low=2, high=10)
        # else:
        #     diff_factor = np.random.uniform(low=10, high=30)
        p = np.random.rand() * 6
        if 0 <= p < 2:
            diff_factor = np.random.uniform(low=0.1, high=0.3)
        elif 2 <= p < 4:
            diff_factor = np.random.uniform(low=0.5, high=2)
        elif 4 <= p < 5:
            diff_factor = np.random.uniform(low=2, high=10)
        else:
            diff_factor = np.random.uniform(low=10, high=30)

        offset_range = np.min(wh1, axis=1) / 2
        dxy = np.random.uniform(low=-offset_range, high=offset_range, size=(1, 2))
        dwh = np.random.uniform(low=-0.02, high=0.02, size=(1, 2))
        dtheta = np.random.uniform(low=-np.pi / 4, high=np.pi / 4, size=(1, 1))
        xy2 = xy1 + dxy / diff_factor
        wh2 = wh1 + dwh / diff_factor
        theta2 = theta1 + dtheta / diff_factor
        rbbox2 = np.concatenate([xy2, wh2, theta2], axis=1)

        poly1 = RotBox2Polys(rbbox1)
        poly2 = RotBox2Polys(rbbox2)
        large_poly1 = poly1 *512 + 512  #放大框，便于obb_overlap的计算
        large_poly2 = poly2 *512 + 512
        rbbox1 = polygonToRotRectangle_batch(large_poly1)
        rbbox2 = polygonToRotRectangle_batch(large_poly2)

        #normalize
        # poly1 = (RotBox2Polys(rbbox1) - 512) / 512
        # poly2 = (RotBox2Polys(rbbox2) - 512) / 512
        #norm_rbbox1 = polygonToRotRectangle_batch(poly1)
        #norm_rbbox2 = polygonToRotRectangle_batch(poly2)
        xmin1, xmax1, ymin1, ymax1 = np.min(poly1[:,0::2]), np.max(poly1[:,0::2]), np.min(poly1[:,1::2]), np.max(poly1[:,1::2])
        xmin2, xmax2, ymin2, ymax2 = np.min(poly2[:,0::2]), np.max(poly2[:,0::2]), np.min(poly2[:,1::2]), np.max(poly2[:,1::2])
        
        if (xmin1 > -1.2) & (xmax1 < 1.2) & (ymin1 > -1.2) & (ymax1 < 1.2) & (xmin2 > -1.2) & (xmax2 < 1.2) & (ymin2 > -1.2) & (ymax2 < 1.2):
            polys1 = np.concatenate([polys1, poly1], axis=0)    #归一化后的8参数框
            polys2 = np.concatenate([polys2, poly2], axis=0)
            rbboxes1 = np.concatenate([rbboxes1, rbbox1], axis=0)   #未归一化的5参数框
            rbboxes2 = np.concatenate([rbboxes2, rbbox2], axis=0)
            count += 1

    polys1 = torch.Tensor(polys1).cuda()
    polys2 = torch.Tensor(polys2).cuda()
    rbboxes1 = torch.Tensor(rbboxes1).cuda()
    rbboxes2 = torch.Tensor(rbboxes2).cuda()

    return polys1, polys2, rbboxes1, rbboxes2






