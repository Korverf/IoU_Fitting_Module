import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
#from torch.nn import functional as F
from tensorboardX import SummaryWriter
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
#from logger import Logger
from progress.bar import Bar

from modules.iou_fit.iou_fit_38_res import get_model
from data_generate.data_generate_new_11 import data_generator
from ops.box_iou_rotated import obb_overlaps


def parse_args():
    parser = argparse.ArgumentParser(description='IOU Fit Module')
    parser.add_argument('--method', type=str, default='iou_fit_module_49_res')
    parser.add_argument('--iters', default=8000000, type=int)
    parser.add_argument('--print_inter', type=int, default=2000,
                        help='disable progress bar and print to screen.')
    parser.add_argument('--save_inter', type=int, default=10000,
                        help='disable progress bar and print to screen.')
    parser.add_argument('--val_inter', type=int, default=10000,
                        help='validation interval.')
    parser.add_argument('--val_num', type=int, default=1000,
                        help='number of samples to validate.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Must be divisible by 4.')
    parser.add_argument('--lr', default=1e-5, type=float)#2.5e-5
    parser.add_argument('--lr_step', type=str, default='-1', help='drop learning rate by 10.')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument('--works_dir', type=str, default='./runs/')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--restore_from', type=str, default='')
    parser.add_argument('--load_from', type=str, default='')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--start-iter', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    args = parser.parse_args()

    return args

def main(args):
    # initialization
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    if not os.path.exists(args.works_dir):
        os.makedirs(args.works_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.works_dir, args.method))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.enabled = True

    iou_fit_module = get_model(in_features=16, hidden_features=16)
    #iou_fit_module = get_model(in_channels=16, inter_channels=512)
    if args.load_from:
        if os.path.isfile(args.load_from):
            checkpoint = torch.load(args.load_from, map_location='cpu')
            iou_fit_module.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(args.load_from))
        else:
            print("=> no checkpoint found at '{}'".format(args.load_from))
    # optimizer = optim.SGD(
    #     [{'params': filter(lambda p: p.requires_grad, iou_fit_module.parameters()), 'lr': args.lr}],
    #     lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(iou_fit_module.parameters(), lr=args.lr)

    iou_fit_module.float()
    iou_fit_module.cuda()

    cudnn.benchmark = True

    if args.restore_from:
        if os.path.isfile(args.restore_from):
            print("=> loading checkpoint '{}'".format(args.restore_from))
            if args.gpu is None:
                checkpoint = torch.load(args.restore_from)
            else:
                # Map model to be loaded to specified single gpu.
                #loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.restore_from, map_location='cuda:0')
                #checkpoint = torch.load(args.restore_from, map_location=loc)
            args.start_iter = checkpoint['epoch']
            iou_fit_module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (iter {})"
                  .format(args.restore_from, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_from))


    start = time.time()
    num_iters = args.iters
    bar = Bar('{}'.format(args.method), max=num_iters)
    best_score_5 = 0.0
    best_score_5_iter = 1
    best_score_1 = 0.0
    best_score_1_iter = 1
    for iter in range(args.start_iter+1, num_iters+1):
        #print('\n{} | {}'.format(iter, num_iters))

        # training
        phase = 'train'
        iou_fit_module.train()
        polys1, polys2, rbboxes1, rbboxes2 = data_generator(args.batch_size)    # B,N,5
        iou_fit_value = iou_fit_module.forward(polys1, polys2)
        iou_fit_value = iou_fit_value[:, 0].clamp(min=1e-6, max=1)
        #iou_fit_value = iou_fit_value.clamp(min=1e-6, max=1)
        loss = iou_fit_module.loss(rbboxes1, rbboxes2, iou_fit_value)
        #loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=iou_fit_module.parameters(), max_norm=35, norm_type=2)
        optimizer.step()
        # tensorboard record
        if iter % 100 == 0:
            #writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], iter)
            writer.add_scalar('iou_fit_loss', loss, iter)
        
        Bar.suffix = '{phase}: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            iter, num_iters, phase=phase,
            total=bar.elapsed_td, eta=bar.eta_td)

        Bar.suffix = Bar.suffix + '|iou_fit_loss {:.4f} '.format(loss)
        if args.print_inter > 0:
            if iter % args.print_inter == 0:
                print('{}| {}'.format(args.method, Bar.suffix))
        else:
            bar.next()

        #validation
        if iter % args.val_inter == 0:
            iou_fit_module.eval()
            phase = 'val'
            print('Validation Stage:')
            total_score_5 = 0.0
            total_score_1 = 0.0
            bar_val = Bar('{}'.format(args.method), max=args.val_num)
            for j in range(1, args.val_num+1):
                polys1, polys2, rbboxes1, rbboxes2 = data_generator(args.batch_size)
                iou_fit_value = iou_fit_module.forward(polys1, polys2)
                #iou_fit_value = iou_fit_value.clamp(min=1e-6, max=1)
                iou_fit_value = iou_fit_value[:, 0].clamp(min=1e-6, max=1)
                IoU_targets = obb_overlaps(rbboxes1, rbboxes2, is_aligned=True).squeeze(
                    1).clamp(min=1e-6, max=1)
                for i in range(len(IoU_targets)):
                    dif = abs(IoU_targets[i] - iou_fit_value[i])
                    if dif < 0.05:
                        total_score_5 += 1
                    if dif < 0.01:
                        total_score_1 += 1
                Bar.suffix = '{phase}: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(j, args.val_num, phase=phase, total=bar_val.elapsed_td, eta=bar_val.eta_td)
                if j % 500 == 0:
                    print('{}| {}'.format(args.method, Bar.suffix))
                
            total_score_5 = (total_score_5 / (args.val_num * args.batch_size)) * 100
            total_score_1 = (total_score_1 / (args.val_num * args.batch_size)) * 100
            if total_score_5 > best_score_5:
                best_score_5 = total_score_5
                best_score_5_iter = iter
            if total_score_1 > best_score_1:
                best_score_1 = total_score_1
                best_score_1_iter = iter
            print('\nscore_0.05: {}'.format(total_score_5))
            print('score_0.01: {}'.format(total_score_1))
            writer.add_scalar('score_0.05', total_score_5, iter)
            writer.add_scalar('score_0.01', total_score_1, iter)

        if (iter % args.save_inter == 0):#(iter > 1000000) &
            model_root = os.path.join(args.works_dir, args.method)
            if not os.path.exists(model_root):
                os.mkdir(model_root)
            model_dir = os.path.join(model_root, 'iter_{:05d}'.format(iter) + '.pth')
            torch.save({
                'epoch': iter,
                'method': args.method,
                'state_dict': iou_fit_module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_dir)
            # torch.save(seg_model.state_dict(), model_dir)
            print('Model saved to %s' % model_dir)

        if iter in args.lr_step:
            lr = args.lr * (0.1 ** (args.lr_step.index(iter) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    bar.finish()
    print('best score_5 is {}, in iter {}'.format(best_score_5, best_score_5_iter))
    print('best score_1 is {}, in iter {}'.format(best_score_1, best_score_1_iter))
    print('Complete using', time.time() - start, 'seconds')


if __name__ == '__main__':
    args = parse_args()
    args.lr_step = [int(i) for i in args.lr_step.split(',')]
    main(args)


