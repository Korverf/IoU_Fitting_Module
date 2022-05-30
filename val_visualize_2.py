import torch
import numpy as np
from box_iou_rotated import obb_overlaps
from iou_fit_38 import get_model
from data_generate_new_11 import data_generator
import os
import cv2
import random
import mmcv
import copy
from transform_rbbox import (polygonToRotRectangle_batch, RotBox2Polys_torch, RotBox2Polys, obb2poly)


def draw_poly_detections(detections, img_shape=(1024, 1024, 3), img=None, showStart=False, colormap=None):
    if img is None:
        img = 255 * np.ones(img_shape, dtype='uint8')
    #color_white = (255, 255, 255)

    if colormap is None:
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
    else:
        color = colormap
    dets = detections

    # 绘制框
    if len(dets) > 0:
        for det in dets:
            bbox = det
            bbox = list(map(int, bbox))
            if showStart:# 在第一个坐标绘制一个小圆点
                cv2.circle(img, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)
            #绘制边框
            for i in range(3):
                cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i+1) * 2], bbox[(i+1) * 2 + 1]),
                            color=color, thickness=1,lineType=cv2.LINE_AA)
            cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=1, lineType=cv2.LINE_AA)
    return img


def main():
    """Create the model and start the evaluation process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    vis_path = '/home/yyw/yyf/Dataset/HRSC2016/vis/vis.jpg'
    poly = torch.tensor([[184, 350, 944, 185, 981, 355, 221, 521]])
    img_path = '/home/yyw/yyf/Dataset/HRSC2016/Train/images/100000001.bmp'
    img = cv2.imread(img_path)
    img = draw_poly_detections(poly, img=img, img_shape=(1166, 753, 3), showStart=False, colormap=(0, 255, 0))

    cv2.imwrite(vis_path, img)





if __name__ == '__main__':
    main()
