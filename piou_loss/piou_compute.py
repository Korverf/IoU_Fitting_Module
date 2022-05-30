import torch
import torch.nn as nn
#from mmdet.ops import box_iou_rotated_differentiable

#from .utils import weighted_loss
from piou.pixel_weights import Pious
# from mmdet.ops import obb_overlaps


def template_w_pixels(width):
  x = torch.tensor(torch.arange(-100, width + 100))
  grid_x = x.float() + 0.5
  return grid_x


# @LOSSES.register_module
class PIoU(nn.Module):

    def __init__(self, img_size=500):
        super(PIoU, self).__init__()
        self.PIoU = Pious(k=10, is_hard=False)
        self.template = template_w_pixels(img_size)

    def forward(self,
                pred,
                target
                ):
        pious = self.PIoU(pred, target, self.template.cuda(pred.get_device())).clamp(min=0.1, max=1.0)

        return pious
