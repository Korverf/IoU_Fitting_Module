import torch
import numpy as np
from box_iou_rotated import obb_overlaps
from iou_fit_conv_1 import get_model
from data_generate_new_2 import data_generator
import os
from transform_rbbox import (polygonToRotRectangle_batch, RotBox2Polys_torch, RotBox2Polys, obb2poly)


def main():
    """Create the model and start the evaluation process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    iou_fit_module = get_model(in_features=16, hidden_features=16)
    restore_from = './runs/iou_fit_module_41/iter_4920000.pth'
    saved_state_dict = torch.load(restore_from)
    iou_fit_module.load_state_dict(saved_state_dict['state_dict'])

    iou_fit_module.eval()
    iou_fit_module.cuda()

    polys1, polys2, rbboxes1, rbboxes2 = data_generator(16)
    #rbboxes1 = polygonToRotRectangle_batch(rbboxes1)
    #rbboxes2 = polygonToRotRectangle_batch(rbboxes2)
    #rbboxes1, rbboxes2 = torch.Tensor(rbboxes1).cuda(), torch.Tensor(rbboxes2).cuda()
    #rbboxes1 = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0.]]).cuda()
    #rbboxes2 = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1.]]).cuda()
    with torch.no_grad():
        #iou_fit_value = iou_fit_module.forward(rbboxes1[:, :], rbboxes2[:, :])
        #iou_fit_value = iou_fit_value[:, 0].clamp(min=1e-6)
        IoU_targets = obb_overlaps(rbboxes1, rbboxes2.detach(), is_aligned=True).squeeze(
            1).clamp(min=1e-6)
    torch.set_printoptions(sci_mode=False, precision=6)
    #print(iou_fit_value)
    print(IoU_targets)



if __name__ == '__main__':
    main()
