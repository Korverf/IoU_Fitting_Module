import torch
from ops.box_iou_rotated import obb_overlaps
from modules.iou_fit.iou_fit_38_res import get_model
from data_generate.data_generate_new_11 import data_generator
import os
from progress.bar import Bar
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='IOU Fit Module')
    parser.add_argument('--method', type=str, default='iou_fit_module_49_res')
    parser.add_argument('--print_inter', type=int, default=200,
                        help='disable progress bar and print to screen.')
    parser.add_argument('--val_num', type=int, default=1000,
                        help='number of samples to validate.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Must be divisible by 4.')
    args = parser.parse_args()

    return args


def main(args):
    """Create the model and start the evaluation process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    iou_fit_module = get_model(in_features=16, hidden_features=16)
    #restore_from = './runs/iou_fit_module_49_rerun/iter_7370000.pth'
    restore_from = './runs/iou_fit_module_49_res/iter_7440000.pth'
    #restore_from = './runs/iou_fit_fc_3_new/iter_2740000.pth'
    saved_state_dict = torch.load(restore_from)
    iou_fit_module.load_state_dict(saved_state_dict['state_dict'])
    iou_fit_module.eval()
    iou_fit_module.cuda()
    #valiation
    phase = 'val'
    print('Validation Stage:')
    total_score_5 = 0.0
    total_score_1 = 0.0
    mean_dif = 0.0
    #total_score_05 = 0.0
    bar_val = Bar('{}'.format(args.method), max=args.val_num)
    for j in range(1, args.val_num+1):
        polys1, polys2, rbboxes1, rbboxes2 = data_generator(args.batch_size)
        torch.set_printoptions(sci_mode=False, precision=6)
        with torch.no_grad():
            iou_fit_value = iou_fit_module.forward(polys1, polys2)
            iou_fit_value = iou_fit_value[:, 0].clamp(min=1e-6, max=1)
            IoU_targets = obb_overlaps(rbboxes1, rbboxes2, is_aligned=True).squeeze(1).clamp(min=1e-6, max=1)
            for i in range(len(IoU_targets)):
                dif = abs(IoU_targets[i] - iou_fit_value[i])
                mean_dif += dif
                if dif < 0.05:
                    total_score_5 += 1
                if dif < 0.01:
                    total_score_1 += 1
                # if dif < 0.005:
                #     total_score_05 += 1
        Bar.suffix = '{phase}: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(j, args.val_num, phase=phase, total=bar_val.elapsed_td, eta=bar_val.eta_td)
        if j % args.print_inter == 0:
            print('{}| {}'.format(args.method, Bar.suffix))

    total_score_5 = (total_score_5 / (args.val_num * args.batch_size)) * 100
    total_score_1 = (total_score_1 / (args.val_num * args.batch_size)) * 100
    mean_dif = mean_dif / (args.val_num * args.batch_size)
    #total_score_05 = (total_score_05 / (args.val_num * args.batch_size)) * 100
    print('\n与实际IOU误差小于0.05的得分: {}'.format(total_score_5))
    print('与实际IOU误差小于0.01的得分: {}'.format(total_score_1))
    print('与实际IoU的平均误差为: ', mean_dif)
    #print('与实际IOU误差小于0.005的得分: {}'.format(total_score_05))

if __name__ == '__main__':
    args = parse_args()
    main(args)
