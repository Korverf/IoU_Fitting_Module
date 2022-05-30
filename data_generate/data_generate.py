import numpy as np
import torch

def polygonToRotRectangle_batch(bbox, with_module=True):
    """
    :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
            shape [num_boxes, 8]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
            shape [num_rot_recs, 5]
    """
    # print('bbox: ', bbox)
    bbox = np.array(bbox,dtype=np.float32)
    bbox = np.reshape(bbox,newshape=(-1, 2, 4),order='F') #(x1,x2,x3,x4) (y1,y2,y3,y4)
    # angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])
    # print('bbox: ', bbox)
    angle = np.arctan2(-(bbox[:, 0, 1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0]) #-(x2-x1)/(y2-y1)
    #angle [-pi,pi]
    # angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
    # center = [[0],[0]] ## shape [2, 1]
    # print('angle: ', angle)
    center = np.zeros((bbox.shape[0], 2, 1))
    for i in range(4):
        center[:, 0, 0] += bbox[:, 0,i]
        center[:, 1, 0] += bbox[:, 1,i]

    center = np.array(center,dtype=np.float32)/4.0
    # R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose((2, 1, 0)),bbox-center)


    xmin = np.min(normalized[:, 0, :], axis=1)
    # print('diff: ', (xmin - normalized[:, 0, 3]))
    # assert sum((abs(xmin - normalized[:, 0, 3])) > eps) == 0
    xmax = np.max(normalized[:, 0, :], axis=1)
    # assert sum(abs(xmax - normalized[:, 0, 1]) > eps) == 0
    # print('diff2: ', xmax - normalized[:, 0, 1])
    ymin = np.min(normalized[:, 1, :], axis=1)
    # assert sum(abs(ymin - normalized[:, 1, 3]) > eps) == 0
    # print('diff3: ', ymin - normalized[:, 1, 3])
    ymax = np.max(normalized[:, 1, :], axis=1)
    # assert sum(abs(ymax - normalized[:, 1, 1]) > eps) == 0
    # print('diff4: ', ymax - normalized[:, 1, 1])

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    w = w[:, np.newaxis]
    h = h[:, np.newaxis]
    # TODO: check it
    if with_module:
        angle = angle[:, np.newaxis] % ( 2 * np.pi)
    else:
        angle = angle[:, np.newaxis]
    dboxes = np.concatenate((center[:, 0].astype(np.float), center[:, 1].astype(np.float), w, h, angle), axis=1)
    return dboxes


def RotBox2Polys(dboxes):
    """
    :param dboxes: (x_ctr, y_ctr, w, h, angle)
        (numboxes, 5)
    :return: quadranlges:
        (numboxes, 8)
    """
    cs = np.cos(dboxes[:, 4])
    ss = np.sin(dboxes[:, 4])
    w = dboxes[:, 2] - 1
    h = dboxes[:, 3] - 1

    ## change the order to be the initial definition
    x_ctr = dboxes[:, 0]
    y_ctr = dboxes[:, 1]
    x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
    x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
    x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)

    y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
    y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
    y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
    y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)

    x1 = x1[:, np.newaxis]
    y1 = y1[:, np.newaxis]
    x2 = x2[:, np.newaxis]
    y2 = y2[:, np.newaxis]
    x3 = x3[:, np.newaxis]
    y3 = y3[:, np.newaxis]
    x4 = x4[:, np.newaxis]
    y4 = y4[:, np.newaxis]

    polys = np.concatenate((x1, y1, x2, y2, x3, y3, x4, y4), axis=1)
    return polys


def RotBox2Polys_torch(dboxes):
    """

    :param dboxes:
    :return:
    """
    cs = torch.cos(dboxes[:, 4])
    ss = torch.sin(dboxes[:, 4])
    w = dboxes[:, 2] - 1
    h = dboxes[:, 3] - 1

    x_ctr = dboxes[:, 0]
    y_ctr = dboxes[:, 1]
    x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
    x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
    x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)

    y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
    y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
    y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
    y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)

    polys = torch.cat((x1.unsqueeze(1),
                       y1.unsqueeze(1),
                       x2.unsqueeze(1),
                       y2.unsqueeze(1),
                       x3.unsqueeze(1),
                       y3.unsqueeze(1),
                       x4.unsqueeze(1),
                       y4.unsqueeze(1)), 1)

    return polys


def data_generator(num):
    #N = np.random.randint(low=1, high=10, size=None, dtype=int)
    N = num
    rbboxes1 = np.zeros((0, 8))
    rbboxes2 = np.zeros((0, 8))
    for i in range(N):
        xy1 = np.random.rand(1, 2) * 1024
        w1 = np.random.uniform(low=10, high=100, size=(1, 1))
        ratio1 = np.random.uniform(low=0.25, high=2.5, size=(1, 1))
        h1 = w1 * ratio1
        theta1 = np.random.uniform(low=-np.pi, high=np.pi, size=(1, 1))
        rbbox1 = np.concatenate([xy1, w1, h1, theta1], axis=1)

        #diff_factor = np.random.uniform(low=0.1, high=0.3)
        #diff_factor = np.random.uniform(low=0.5, high=2)
        diff_factor = np.random.uniform(low=10, high=30)
        # p = np.random.rand() * 5
        # if 0 <= p < 2:
        #     diff_factor = np.random.uniform(low=0.1, high=0.3)
        # elif 2 <= p < 3:
        #     diff_factor = np.random.uniform(low=0.5, high=2)
        # elif 3 <= p < 4:
        #     diff_factor = np.random.uniform(low=2, high=10)
        # else:
        #     diff_factor = np.random.uniform(low=10, high=30)
        # p = np.random.rand() * 6
        # if 0 <= p < 2:
        #     diff_factor = np.random.uniform(low=0.1, high=0.3)
        # elif 2 <= p < 4:
        #     diff_factor = np.random.uniform(low=0.5, high=2)
        # elif 4 <= p < 5:
        #     diff_factor = np.random.uniform(low=2, high=10)
        # else:
        #     diff_factor = np.random.uniform(low=10, high=30)

        offset_range = np.min([w1, h1], axis=0) / 2
        dxy = np.random.uniform(low=-offset_range, high=offset_range, size=(1, 2))
        dw = np.random.uniform(low=-10, high=10, size=(1, 1))
        dratio = np.random.uniform(low=-0.25, high=0.25, size=(1, 1))
        dtheta = np.random.uniform(low=-np.pi / 4, high=np.pi / 4, size=(1, 1))

        xy2 = xy1 + dxy / diff_factor
        w2 = w1 + dw / diff_factor
        ratio2 = ratio1 + dratio / diff_factor
        h2 = w2 * ratio2
        theta2 = theta1 + dtheta / diff_factor
        rbbox2 = np.concatenate([xy2, w2, h2, theta2], axis=1)

        #normalize
        poly1 = RotBox2Polys(rbbox1) / 1024
        poly2 = RotBox2Polys(rbbox2) / 1024
        #rbbox1 = polygonToRotRectangle_batch(poly1)
        #rbbox2 = polygonToRotRectangle_batch(poly2)
        rbboxes1 = np.concatenate([rbboxes1, poly1], axis=0)
        rbboxes2 = np.concatenate([rbboxes2, poly2], axis=0)


    rbboxes1 = torch.Tensor(rbboxes1).cuda()
    rbboxes2 = torch.Tensor(rbboxes2).cuda()

    return rbboxes1, rbboxes2






