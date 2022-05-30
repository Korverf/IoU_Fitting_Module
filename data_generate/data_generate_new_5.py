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
        p1 = np.random.rand() * 3  #确定目标大小
        p2 = np.random.rand() * 4 #确定box2与box1的偏差
        dtheta = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(1, 1))
        if 0 <= p1 < 1: #小
            xy1 = 2 * np.random.rand(1, 2) - 1  #(-1,1)
            wh1 = np.random.uniform(low=5e-3, high=0.04, size=(1, 2))
            theta1 = np.random.uniform(low=-np.pi, high=np.pi, size=(1, 1))
            if 0 <= p2 < 2:
                dxy = np.random.uniform(low=-np.min(wh1, axis=1) / 2, high=np.min(wh1, axis=1) / 2, size=(1, 2))
                xy2 = xy1 + dxy / np.random.uniform(low=0.5, high=2)
                wh2 = wh1 * np.random.uniform(low=0.8, high=1.2, size=(1, 2))
                theta2 = theta1 + dtheta / np.random.uniform(low=2, high=10)
            else:
                dxy = np.random.uniform(low=-np.min(wh1, axis=1) / 2, high=np.min(wh1, axis=1) / 2, size=(1, 2))
                xy2 = xy1 + dxy / np.random.uniform(low=10, high=30)
                wh2 = wh1 * np.random.uniform(low=0.8, high=1.2, size=(1, 2))
                theta2 = theta1 + dtheta / np.random.uniform(low=5, high=30)
        elif 1 <= p1 < 2: #中
            xy1 =  2 * np.random.rand(1, 2) - 1  #(-1,1)
            wh1 = np.random.uniform(low=0.04, high=0.2, size=(1, 2))
            theta1 = np.random.uniform(low=-np.pi, high=np.pi, size=(1, 1))
            if 0 <= p2 < 1:
                xy2 =  2 * np.random.rand(1, 2) - 1  #(-1,1)
                wh2 = wh1 * np.random.uniform(low=0.5, high=2.0, size=(1, 2))
                theta2 = theta1 + dtheta
            elif 1 <= p2 < 3:
                dxy = np.random.uniform(low=-np.min(wh1, axis=1) / 2, high=np.min(wh1, axis=1) / 2, size=(1, 2))
                xy2 = xy1 + dxy / np.random.uniform(low=0.5, high=2)
                wh2 = wh1 * np.random.uniform(low=0.8, high=1.2, size=(1, 2))
                theta2 = theta1 + dtheta / np.random.uniform(low=2, high=10)
            else:
                dxy = np.random.uniform(low=-np.min(wh1, axis=1) / 2, high=np.min(wh1, axis=1) / 2, size=(1, 2))
                xy2 = xy1 + dxy / np.random.uniform(low=10, high=30)
                wh2 = wh1 * np.random.uniform(low=0.8, high=1.2, size=(1, 2))
                theta2 = theta1 + dtheta / np.random.uniform(low=5, high=30)
        else: #大
            xy1 =  np.random.uniform(low=-0.5, high=0.5, size=(1, 2))
            wh1 = np.random.uniform(low=0.2, high=1.6, size=(1, 2))
            theta1 = np.random.uniform(low=-np.pi, high=np.pi, size=(1, 1))
            if 0 <= p2 < 2:
                dxy = np.random.uniform(low=-np.min(wh1, axis=1) / 2, high=np.min(wh1, axis=1) / 2, size=(1, 2))
                xy2 = xy1 + dxy / np.random.uniform(low=0.5, high=2)
                wh2 = wh1 * np.random.uniform(low=0.8, high=1.2, size=(1, 2))
                theta2 = theta1 + dtheta / np.random.uniform(low=2, high=10)
            else:
                dxy = np.random.uniform(low=-np.min(wh1, axis=1) / 2, high=np.min(wh1, axis=1) / 2, size=(1, 2))
                xy2 = xy1 + dxy / np.random.uniform(low=10, high=30)
                wh2 = wh1 * np.random.uniform(low=0.8, high=1.2, size=(1, 2))
                theta2 = theta1 + dtheta / np.random.uniform(low=5, high=30)
        #theta1 = np.random.uniform(low=-np.pi, high=np.pi, size=(1, 1))
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
        # p2 = np.random.rand() * 4 #确定box2与box1的偏差
        # dtheta = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(1, 1))
        # if 0 <= p2 < 1:
        #     if p1 < 2 :
        #         xy2 =  2 * np.random.rand(1, 2) - 1  #(-1,1)
        #     else:
        #         xy2 =  np.random.uniform(low=-0.5, high=0.5, size=(1, 2))
        #     wh2 = wh1 * np.random.uniform(low=0.5, high=2.0, size=(1, 2))
        #     theta2 = theta1 + dtheta
        # elif 1 <= p2 < 2:
        #     dxy = np.random.uniform(low=-np.min(wh1, axis=1) / 2, high=np.min(wh1, axis=1) / 2, size=(1, 2))
        #     xy2 = xy1 + dxy / np.random.uniform(low=2, high=10)
        #     wh2 = wh1 * np.random.uniform(low=0.8, high=1.2, size=(1, 2))
        #     theta2 = theta1 + dtheta / np.random.uniform(low=2, high=10)
        # else:
        #     dxy = np.random.uniform(low=-np.min(wh1, axis=1) / 2, high=np.min(wh1, axis=1) / 2, size=(1, 2))
        #     xy2 = xy1 + dxy / np.random.uniform(low=10, high=30)
        #     wh2 = wh1 * np.random.uniform(low=0.8, high=1.2, size=(1, 2))
        #     theta2 = theta1 + dtheta / np.random.uniform(low=5, high=30)
        
        if wh2[0][0] < 2e-3 or wh2[0][0] > 1.6:
            wh2[0][0] = wh1[0][0]
        if wh2[0][1] < 2e-3 or wh2[0][1] > 1.6:
            wh2[0][1] = wh1[0][1]
        theta2[0][0] = min(theta2[0][0], np.pi)
        theta2[0][0] = max(theta2[0][0], -np.pi)

        # p3 = np.random.rand() * 4 #确定wh2与wh1的偏差
        # if 0 <= p3 < 2:
        #     wh2 = wh1 * np.random.uniform(low=0.8, high=1.2, size=(1, 2))
        # elif 2 <= p3 < 3:
        #     wh2 = wh1 * np.random.uniform(low=0.4, high=0.8, size=(1, 2))
        # else:
        #     wh2 = wh1 * np.random.uniform(low=1.2, high=2.0, size=(1, 2))
        # if wh2[0][0] < 2e-3 or wh2[0][0] > 1.6:
        #     wh2[0][0] = wh1[0][0]
        # if wh2[0][1] < 2e-3 or wh2[0][1] > 1.6:
        #     wh2[0][1] = wh1[0][1]
        # p4 = np.random.rand() * 4 #确定theta2与theta1的偏差
        
        # if 0 <= p4 < 2:
        #     theta2 = theta1 + dtheta / np.random.uniform(low=10, high=30)
        # elif 2 <= p4 < 3:
        #     theta2 = theta1 + dtheta / np.random.uniform(low=2, high=10)
        # else:
        #     theta2 = theta1 + dtheta / np.random.uniform(low=1, high=2)
        # theta2[0][0] = min(theta2[0][0], np.pi)
        # theta2[0][0] = max(theta2[0][0], -np.pi)

        # offset_range = np.min(wh1, axis=1) / 2
        # dxy = np.random.uniform(low=-offset_range, high=offset_range, size=(1, 2))
        # dwh = np.random.uniform(low=0.5, high=2, size=(1, 2))
        # dtheta = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(1, 1))
        # xy2 = xy1 + dxy / diff_factor
        # wh2 = wh1 * dwh

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






