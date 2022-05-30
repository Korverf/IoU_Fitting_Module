import torch
import numpy as np
from box_iou_rotated import obb_overlaps
from iou_fit_conv_1 import get_model
from data_generate_new_11 import data_generator
import os
import cv2
import random
# import mmcv
# import copy
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
    iou_fit_module = get_model(in_features=16, hidden_features=16)
    #restore_from = './runs/iou_fit_module_48/iter_1780000.pth'
    restore_from = './runs/iou_fit_conv_1/iter_4990000.pth'
    #restore_from = './runs/iou_fit_module_42/iter_4750000.pth'
    saved_state_dict = torch.load(restore_from)
    iou_fit_module.load_state_dict(saved_state_dict['state_dict'])
    vis_path = '/home/yyw/yyf/projects/iou_fit_module/vis/vis.jpg'
    vis_path2 = '/home/yyw/yyf/projects/iou_fit_module/vis/vis2.jpg'

    iou_fit_module.eval()
    iou_fit_module.cuda()
    # rbboxes11 = torch.tensor([[273.971069, 137.888901, 106., 260.,  -1.566965],
    #                           [511.501923, 226.539078,  33.456219, 114.521751,  -1.568778]]).cuda()
    # rbboxes22 = torch.tensor([[273.971069, 137.888901,  106., 260.,  -0.078235],
    #                           [511.339478, 226.701828,  33.179398, 114.651840,  -1.479340]]).cuda()
    # rbboxes11 = torch.tensor([[273.971069, 137.888901, 106., 260.,  -1.566965],
    #                           [271.971069, 126.888901,  103., 271.,  -1.278235]]).cuda()
    # rbboxes22 = torch.tensor([[271.971069, 126.888901,  103., 271.,  -1.278235],
    #                           [273.971069, 137.888901, 106., 260.,  -1.566965]]).cuda()
    # polys11 = np.array([[301., 142., 365., 142., 365., 372., 300., 371.]])
    # polys22 = np.array([[332., 143.,  366., 148., 334., 371., 299., 366.]])
    # rbboxes11 = torch.tensor(polygonToRotRectangle_batch(polys11)).float().cuda()
    # rbboxes22 = torch.tensor(polygonToRotRectangle_batch(polys22)).float().cuda()
    # pos_poly_pred_decode, pos_poly_target_decode = RotBox2Polys_torch(rbboxes11), RotBox2Polys_torch(rbboxes22)
    # polys1_draw = copy.deepcopy(pos_poly_pred_decode)
    # polys2_draw = copy.deepcopy(pos_poly_target_decode)
    # # 归一化
    # x = torch.cat([pos_poly_pred_decode[:, 0::2], pos_poly_target_decode[:, 0::2]], dim=1)  # N,8
    # xmin, _ = torch.min(x, dim=1)
    # xmax, _ = torch.max(x, dim=1)
    # y = torch.cat([pos_poly_pred_decode[:, 1::2], pos_poly_target_decode[:, 1::2]], dim=1)  # N,8
    # ymin, _ = torch.min(y, dim=1)  # N
    # ymax, _ = torch.max(y, dim=1)
    # enclose_bbox_point1 = torch.cat([xmin.unsqueeze(-1), ymin.unsqueeze(-1)], dim=1).detach().cpu().numpy()
    # enclose_bbox_point1 = enclose_bbox_point1.astype(int)
    # enclose_bbox_point2 = torch.cat([xmax.unsqueeze(-1), ymax.unsqueeze(-1)], dim=-1).detach().cpu().numpy()
    # enclose_bbox_point2 = enclose_bbox_point2.astype(int)
    # w = torch.sub(xmax, xmin).unsqueeze(-1)
    # h = torch.sub(ymax, ymin).unsqueeze(-1)
    # long_side, _ = torch.max(torch.cat([w, h], dim=1), dim=1)  # N
    # pos_poly_pred_decode[:, 0::2] = pos_poly_pred_decode[:, 0::2] - xmin.unsqueeze(-1)
    # pos_poly_pred_decode[:, 1::2] = pos_poly_pred_decode[:, 1::2] - ymin.unsqueeze(-1)
    # pos_poly_pred_decode = pos_poly_pred_decode / long_side.unsqueeze(-1)
    #
    # pos_poly_target_decode[:, 0::2] = torch.sub(pos_poly_target_decode[:, 0::2], xmin.unsqueeze(-1))
    # pos_poly_target_decode[:, 1::2] = torch.sub(pos_poly_target_decode[:, 1::2], ymin.unsqueeze(-1))
    # pos_poly_target_decode = pos_poly_target_decode / long_side.unsqueeze(-1)
    #
    # # [0,1] -> [-1,1]
    # # pos_poly_pred_decode_new = 2 * pos_poly_pred_decode - 1
    # # pos_poly_target_decode_new = 2 * pos_poly_target_decode - 1
    # polys1, polys2 = pos_poly_pred_decode, pos_poly_target_decode
    #
    # # polys1, polys2 = (polys1_draw - 400) / 400, (polys2_draw - 400) / 400
    # polys1_draw = polys1_draw.detach().cpu().numpy()
    # polys2_draw = polys2_draw.detach().cpu().numpy()
    # polys1_draw_2 = pos_poly_pred_decode * 500
    # polys2_draw_2 = pos_poly_target_decode * 500
    # polys1_draw_2 = polys1_draw_2.detach().cpu().numpy()
    # polys2_draw_2 = polys2_draw_2.detach().cpu().numpy()
    # with torch.no_grad():
    #     iou_fit_value = iou_fit_module.forward(polys1, polys2)
    #     iou_fit_value = iou_fit_value[:, 0].clamp(min=1e-6, max=1)

    #polys1, polys2, rbboxes11, rbboxes22, enclose_bbox_point1, enclose_bbox_point2 = data_generator(32)
    polys1, polys2, rbboxes11, rbboxes22 = data_generator(64)
    with torch.no_grad():
        torch.set_printoptions(sci_mode=False, precision=6)
        iou_fit_value = iou_fit_module.forward(polys1, polys2)
        iou_fit_value = iou_fit_value[:, 0].clamp(min=1e-6, max=1)

    colormap = [
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 0)]

    polys1 = polys1.detach().cpu().numpy()
    polys2 = polys2.detach().cpu().numpy()
    polys1_draw_2 = (polys1 * 500)
    polys2_draw_2 = (polys2 * 500)
    polys1_draw, polys2_draw = RotBox2Polys_torch(rbboxes11), RotBox2Polys_torch(rbboxes22)
    polys1_draw = polys1_draw.detach().cpu().numpy()
    polys2_draw = polys2_draw.detach().cpu().numpy()

    # img = draw_poly_detections(polys1_draw[0:2], img_shape=(1000, 1000, 3), showStart=False, colormap=colormap[0])
    # img = draw_poly_detections(polys2_draw[0:2], img_shape=(1000, 1000, 3), img=img, showStart=False, colormap=colormap[1])
    # cv2.rectangle(img, enclose_bbox_point1[0], enclose_bbox_point2[0], colormap[2], 1)
    # cv2.imwrite(vis_path, img)
    #
    # img2 = draw_poly_detections(polys1_draw_2[0:2], img_shape=(500, 500, 3), showStart=False, colormap=colormap[0])
    # img2 = draw_poly_detections(polys2_draw_2[0:2], img_shape=(500, 500, 3), img=img2, showStart=False, colormap=colormap[1])
    #
    # cv2.imwrite(vis_path2, img2)

    #rbboxes1, rbboxes2 = torch.Tensor(rbboxes1).cuda(), torch.Tensor(rbboxes2).cuda()
    with torch.no_grad():
        # IoU_targets = obb_overlaps(rbboxes1, rbboxes2.detach(), is_aligned=True).squeeze(
        #     1).clamp(min=1e-6)
        IoU_targets2 = obb_overlaps(rbboxes11, rbboxes22.detach(), is_aligned=True).squeeze(
            1).clamp(min=1e-6)

    torch.set_printoptions(sci_mode=False, precision=6)
    print('box1:', polys1_draw)
    #print('box2:', polys2_draw)
    print('fit iou:', iou_fit_value)
    print('real iou target', IoU_targets2)
    #print('normalize iou target:', IoU_targets)




if __name__ == '__main__':
    main()
