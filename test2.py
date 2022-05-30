import torch
import numpy as np
from box_iou_rotated import obb_overlaps



if __name__ == '__main__':
    N = np.random.randint(low=1, high=10, size=None, dtype=int)
    xy1 = np.random.rand(N, 2) * 1024
    w1 = np.random.uniform(low=10, high=100, size=(N, 1))
    ratio1 = np.random.uniform(low=0.25, high=2.5, size=(N, 1))
    h1 = w1 * ratio1
    theta1 = np.random.uniform(low=-np.pi, high=np.pi, size=(N, 1))
    rbboxes1 = np.concatenate([xy1, w1, h1, theta1], axis=1)

    p = np.random.rand() * 5
    if 0 <= p < 2:
        diff_factor = np.random.uniform(low=0.1, high=0.3)
    elif 2 <= p < 3:
        diff_factor = np.random.uniform(low=0.5, high=2)
    elif 3 <= p < 4:
        diff_factor = np.random.uniform(low=2, high=10)
    else:
        diff_factor = np.random.uniform(low=10, high=30)

    offset_range = np.min([w1, h1], axis=0) / 2
    dxy = np.random.uniform(low=-offset_range, high=offset_range, size=(N, 2))

    dw = np.random.uniform(low=-10, high=10, size=(N, 1))

    dratio = np.random.uniform(low=-0.25, high=0.25, size=(N, 1))

    dtheta = np.random.uniform(low=-np.pi/4, high=np.pi/4, size=(N, 1))

    xy2 = xy1 + dxy / diff_factor
    w2 = w1 + dw / diff_factor
    ratio2 = ratio1 + dratio / diff_factor
    h2 = w2 * ratio2
    theta2 = theta1 + dtheta / diff_factor
    rbboxes2 = np.concatenate([xy2, w2, h2, theta2], axis=1)

    IoU_targets = obb_overlaps(rbboxes1[:,:], rbboxes2[:,:], is_aligned=True)
    print(IoU_targets)