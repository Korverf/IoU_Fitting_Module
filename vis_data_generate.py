import torch
from ops.box_iou_rotated import obb_overlaps
from data_generate.data_generate_new_12 import data_generator
import os
from progress.bar import Bar
import argparse
#from statistics import mean
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='IOU Fit Module')
    parser.add_argument('--method', type=str, default='iou_fit_module_49_res')
    parser.add_argument('--print_inter', type=int, default=100,
                        help='disable progress bar and print to screen.')
    parser.add_argument('--val_num', type=int, default=100,
                        help='number of samples to validate.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Must be divisible by 4.')
    args = parser.parse_args()

    return args

def generate_axes(lists, interval, max_value):
    index = list(np.arange(interval/2, max_value+interval/2, interval))
    cut = pd.cut(lists, index)
    counts = pd.value_counts(cut).sort_index()

    x = np.arange(interval, max_value, interval)
    x = np.around(x, 1)
    x = [str(i) for i in x]
    y = counts.values

    return x, y

def main(args):
    phase = 'vis'
    iou_list = np.empty([0,1], dtype=float)
    #mean_iou = 0
    bar_val = Bar('{}'.format(args.method), max=args.val_num)
    for j in range(1, args.val_num+1):
        polys1, polys2, rbboxes1, rbboxes2 = data_generator(args.batch_size)
        torch.set_printoptions(sci_mode=False, precision=6)
        IoU_targets = obb_overlaps(rbboxes1, rbboxes2, is_aligned=True).squeeze(1).clamp(min=1e-6, max=1).cpu().numpy()
        for i in range(len(IoU_targets)):
            iou_list = np.append(iou_list, [IoU_targets[i]])

        Bar.suffix = '{phase}: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(j, args.val_num, phase=phase, total=bar_val.elapsed_td, eta=bar_val.eta_td)
        if j % args.print_inter == 0:
            print('{}| {}'.format(args.method, Bar.suffix))

    mean_iou = np.mean(iou_list)
    print('\n平均IoU: {}'.format(mean_iou))

    #画出IoU分布统计直方图 方法1
    # bins = 10#[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # ranges = (0,1)
    # plt.figure()#figsize=(10, 10)
    # nums, bins, patches = plt.hist(x=iou_list,bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],edgecolor='k',density=True)#,range=(0,1)
    # plt.xticks(bins, bins)
    # # for num, bin in zip(nums, bins):
    # #     num = num / 10
    # #     plt.annotate(num, xy=(bin, num), xytext=(bin + 1.5, num + 0.5))
    # plt.show()

    #画出IoU分布统计直方图 方法2
    # 这两行代码解决 plt 中文显示的问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    #plt.figure()
    # nums, bins, patches = plt.hist(x=iou_list, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], edgecolor='k',
    #                                density=True)  # ,range=(0,1)
    # nums = nums / 10
    nums = [0.76906, 0.12203, 0.05531, 0.03203, 0.01391, 0.00656, 0.00063, 0.00016, 0.00031, 0.00000]
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    x_label = ['[0,0.1]', '(0.1,0.2]', '(0.2,0.3]', '(0.3,0.4]', '(0.4,0.5]', '(0.5,0.6]', '(0.6,0.7]', '(0.7,0.8]', '(0.8,0.9]', '(0.9,1.0]']
    plt.xticks(x, x_label)
    plt.bar(x, nums, width=1.0,bottom=0,align='center',edgecolor='black')
    for i in x:
        plt.text(i,nums[i]+0.01,round(nums[i],2),ha='center',fontsize=8,family='Calibri')
    #plt.title('随机生成策略IoU分布图')
    plt.show()

    # x_interval = 0.1  # 调整width间隔
    # y_interval = 0.1  # 调整height间隔
    # x, y = generate_axes(nums, x_interval, 10)
    # y = [item / len(annotations) for item in y]
    #
    # axs[1].bar(x, y)
    # axs[1].set_title('width(interval= %d )' % width_interval, fontsize=15)
    # axs[1].yaxis.set_major_formatter(PercentFormatter(1))  # 纵轴显示百分数


if __name__ == '__main__':
    args = parse_args()
    main(args)
