import torch
import numpy as np
from ops.box_iou_rotated import obb_overlaps
from modules.iou_fit.iou_fit_38 import get_model
from data_generate.data_generate_new_11 import data_generator
import os
from transform_rbbox import (polygonToRotRectangle_batch, RotBox2Polys_torch, RotBox2Polys, obb2poly)
import cv2
import random


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
            if showStart:
                cv2.circle(img, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)# 在第一个坐标绘制一个小红圆点
                cv2.circle(img, (bbox[2], bbox[3]), 3, (0, 255, 0), -1)# 在第二个坐标绘制一个小绿圆点
                cv2.circle(img, (bbox[4], bbox[5]), 3, (255, 0, 0), -1)  # 在第三个坐标绘制一个小蓝圆点
                cv2.circle(img, (bbox[6], bbox[7]), 3, (0, 0, 0), -1)  # 在第四个坐标绘制一个小黑圆点
            #绘制边框
            for i in range(3):
                cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i+1) * 2], bbox[(i+1) * 2 + 1]),
                            color=color, thickness=1,lineType=cv2.LINE_AA)
            cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=1, lineType=cv2.LINE_AA)
    return img


def main():
    """Create the model and start the evaluation process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    iou_fit_module = get_model(in_features=16, hidden_features=16)
    restore_from = './runs/iou_fit_module_49/iter_1740000.pth'
    saved_state_dict = torch.load(restore_from)
    iou_fit_module.load_state_dict(saved_state_dict['state_dict'])
    vis_path = '/media/yyw/607ca214-10cb-4af6-af97-eac656e135d1/yyw/yyf/projects/iou_fit_module/vis/vis.jpg'
    vis_path2 = '/media/yyw/607ca214-10cb-4af6-af97-eac656e135d1/yyw/yyf/projects/iou_fit_module/vis/boundary_discontinunity/(4)/vis2.jpg'

    colormap = [
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 0)]

    iou_fit_module.eval()
    iou_fit_module.cuda()



    # rbboxes1 = torch.Tensor([[332.7500, 256.7500, 230.0000,  65.0000,  -1.5708]]).cuda()
    # rbboxes2 = torch.Tensor([[332.7500, 257.0000, 225.0000,  55.0001,  -1.4200]]).cuda()
    #默认是计算与ｙ轴的夹角，
    # (2)
    # # rbboxes1 = torch.Tensor([[300., 300., 70.0000, 130.0000, 0.0]]).cuda()
    # # rbboxes2 = torch.Tensor([[295., 305., 65.0000, 132.0000, -0.102]]).cuda()
    # # polys1 = RotBox2Polys_torch(rbboxes1)#tensor([[335., 235., 335., 365., 265., 365., 265., 235.]])
    # # polys2 = RotBox2Polys_torch(rbboxes2)#tensor([[320.6107, 236.0338, 334.0514, 367.3477, 269.3893, 373.9662, 255.9486,　242.6523]])
    # polys1_np = np.array([[265., 235., 335., 235., 335., 365., 265., 365.]])
    # polys2_np = np.array([[269.3893, 373.9662, 255.9486, 242.6523, 320.6107, 236.0338, 334.0514, 367.3477]])
    # rbboxes1 = polygonToRotRectangle_batch(polys1_np, with_module=False)
    # #rbboxes1 = torch.Tensor(rbboxes1).cuda()
    # rbboxes2 = polygonToRotRectangle_batch(polys2_np, with_module=False)
    # rbboxes1, rbboxes2 = torch.Tensor(rbboxes1).cuda(), torch.Tensor(rbboxes2).cuda()
    # polys1 = torch.Tensor(polys1_np).cuda()
    # polys2 = torch.Tensor(polys2_np).cuda()

    # (3)
    # rbboxes1 = torch.Tensor([[300., 300., 110.0000, 130.0000, 0.0]]).cuda()
    # rbboxes2 = torch.Tensor([[295., 305., 115.0000, 128.0000, -0.127]]).cuda()
    # polys1 = RotBox2Polys_torch(rbboxes1)#tensor([[335., 235., 335., 365., 265., 365., 265., 235.]])
    # polys2 = RotBox2Polys_torch(rbboxes2)#tensor([[320.6107, 236.0338, 334.0514, 367.3477, 269.3893, 373.9662, 255.9486,　242.6523]])
    # polys1_np = np.array([[245., 235., 355., 235., 355., 365., 245., 365.]])
    # polys2_np = np.array([[343.9308, 234.2325, 360.1431, 361.2017, 246.0693, 375.7675, 229.8569, 248.7983]])
    # rbboxes1 = polygonToRotRectangle_batch(polys1_np, with_module=False)
    # # rbboxes1 = torch.Tensor(rbboxes1).cuda()
    # rbboxes2 = polygonToRotRectangle_batch(polys2_np, with_module=False)
    # rbboxes1, rbboxes2 = torch.Tensor(rbboxes1).cuda(), torch.Tensor(rbboxes2).cuda()
    # polys1 = torch.Tensor(polys1_np).cuda()
    # polys2 = torch.Tensor(polys2_np).cuda()

    # img = draw_poly_detections(polys1[0:2], img_shape=(500, 500, 3), showStart=True, colormap=colormap[0])
    # img = draw_poly_detections(polys2[0:2], img_shape=(500, 500, 3), img=img, showStart=True, colormap=colormap[1])
    # cv2.imwrite(vis_path, img)

    # (4)
    # rbboxes1 = torch.Tensor([[300., 300., 110.0000, 130.0000, 0.0]]).cuda()
    # rbboxes2 = torch.Tensor([[305., 305., 115.0000, 128.0000, 0.127]]).cuda()
    # polys1 = RotBox2Polys_torch(rbboxes1)
    # polys2 = RotBox2Polys_torch(rbboxes2)
    polys1_np = np.array([[245., 235., 355., 235., 355., 365., 245., 365.]])
    polys2_np = np.array([[239.8569, 361.2017, 256.0692, 234.2325, 370.1431, 248.7983, 353.9308, 375.7675]])
    rbboxes1 = polygonToRotRectangle_batch(polys1_np, with_module=False)
    # rbboxes1 = torch.Tensor(rbboxes1).cuda()
    rbboxes2 = polygonToRotRectangle_batch(polys2_np, with_module=False)
    rbboxes1, rbboxes2 = torch.Tensor(rbboxes1).cuda(), torch.Tensor(rbboxes2).cuda()
    polys1 = torch.Tensor(polys1_np).cuda()
    polys2 = torch.Tensor(polys2_np).cuda()

    # img = draw_poly_detections(polys1[0:2], img_shape=(500, 500, 3), showStart=True, colormap=colormap[0])
    # img = draw_poly_detections(polys2[0:2], img_shape=(500, 500, 3), img=img, showStart=True, colormap=colormap[1])
    # cv2.imwrite(vis_path, img)

    # 归一化
    x = torch.cat([polys1[:, 0::2], polys2[:, 0::2]], dim=1)  # N,8
    xmin, _ = torch.min(x, dim=1)
    xmax, _ = torch.max(x, dim=1)
    y = torch.cat([polys1[:, 1::2], polys2[:, 1::2]], dim=1)  # N,8
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

    polys2[:, 0::2] = torch.sub(polys2[:, 0::2], xmin.unsqueeze(-1))
    polys2[:, 1::2] = torch.sub(polys2[:, 1::2], ymin.unsqueeze(-1))
    # pos_poly_target_decode_new = torch.zeros_like(pos_poly_target_decode)
    polys2 = polys2 / long_side.unsqueeze(-1)

    polys1_draw_2 = (polys1 * 500)
    polys2_draw_2 = (polys2 * 500)
    img2 = draw_poly_detections(polys1_draw_2[0:2], img_shape=(500, 500, 3), showStart=True, colormap=colormap[0])
    img2 = draw_poly_detections(polys2_draw_2[0:2], img_shape=(500, 500, 3), img=img2, showStart=True, colormap=colormap[1])

    cv2.imwrite(vis_path2, img2)


    with torch.no_grad():
        iou_fit_value = iou_fit_module.forward(polys1, polys2)
        #iou_fit_value = iou_fit_value[:, 0].clamp(min=1e-6)
        IoU_targets = obb_overlaps(rbboxes1, rbboxes2.detach(), is_aligned=True).squeeze(
            1).clamp(min=1e-6)
    torch.set_printoptions(sci_mode=False, precision=6)
    print(iou_fit_value)
    print(IoU_targets)



if __name__ == '__main__':
    main()
