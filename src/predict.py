import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import Model
from data import ImageList
import logging
import cPickle as pickle

import numpy as np
import imageio as io

import shapely
import pandas as pd

#model_names = sorted(name for name in models.__dict__
#    if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=24, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-g', '--gpu', default='0', type=str,
                    metavar='G', help='GPUS (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

def main():
    global args, best_prec1, best_loss
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # create model
    #model = torch.nn.DataParallel(Model()).cuda()
    model = Model().cuda()

    #inputs = torch.autograd.Variable(torch.randn(2, 3, 512, 512))
    #model = Model()
    #outputs = model(inputs)
    #print(outputs.size())
    #exit(0)

    #model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    k = 4
    n = 2*k + 1
    args.arch = 'alex'
    args.data = '/home/thuyen/Research/pupil/input/'

    valdir = args.data

    df = pd.read_csv('valid_info.csv')

    valid_loader = torch.utils.data.DataLoader(
        ImageList(df, valdir, for_train=False),
        batch_size=16, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    outputs = []
    for j, (input, target) in enumerate(valid_loader):

        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        output_var = model(input_var)
        outputs.append(output_var.data.cpu().numpy())
        #outputs.append(output_var.data.cpu().numpy() > 0.5)
    outputs = np.concatenate(outputs)
    np.save('preds_raw.npy', outputs)


if __name__ == '__main__':
    main()
