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
import random
import pandas as pd


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-g', '--gpu', default='0', type=str,
                    metavar='G', help='GPUS (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
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
parser.add_argument('--ckpts', dest='ckpts', default='ckpts/')
parser.add_argument('--seed', dest='seed', default=2017, type=int)
parser.add_argument('--data', dest='data')
parser.add_argument('--train_list', dest='train_list')
parser.add_argument('--valid_list', dest='valid_list')
parser.add_argument('--out_file', dest='out_file', default='preds.npy')




log_file = os.path.join("log.txt")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename=log_file)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger('').addHandler(console)

def main():
    global args, best_loss
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(args.seed)
    if not os.path.exists(args.ckpts):
        os.makedirs(args.ckpts)

    # create model
    model = Model().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        df = pd.read_csv(args.valid_list)
        valid_loader = torch.utils.data.DataLoader(
            ImageList(df, args.data, for_train=False),
            batch_size=16, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        outputs = []
        for j, (input, target) in enumerate(valid_loader):

            input_var = torch.autograd.Variable(input.cuda(), volatile=True)
            output_var = model(input_var)
            outputs.append(output_var.data.cpu().numpy())
            #outputs.append(output_var.data.cpu().numpy() > 0.5)
        outputs = np.concatenate(outputs)
        np.save(args.out_file, outputs)
        return



    df = pd.read_csv(args.train_list)

    train_loader = torch.utils.data.DataLoader(
        ImageList(df, args.data, for_train=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)


    eps = 1e-2

    def criterion(x, y):
        num = 2*(x*y).sum() + eps
        den = x.sum() + y.sum() + eps
        return -num/den


    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)


    logging.info('-------------- New training session, LR = %f ----------------' % (args.lr, ))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch) # adam, same lr

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        ## evaluate on validation set
        #valid_loss = validate(valid_loader, model, criterion)

        is_best = False
        filename = os.path.join(args.ckpts, 'model_{}.pth.tar'.format(epoch+1))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict()
        }, is_best, filename=filename)

        msg = 'Epoch: {0:02d} Train loss {1:.4f}'.format(epoch+1, train_loss)
        logging.info(msg)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg


def validate(valid_loader, model, criterion1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    targets = []
    outputs =[]

    eps = 1e-2
    def criterion(x, y):
        x = x.transpose(1, 0, 2, 3).reshape(10, -1)
        y = y.transpose(1, 0, 2, 3).reshape(10, -1)
        num = (x*y).sum(1) + eps
        den = x.sum(1) + y.sum(1) - num + 2*eps
        return -np.mean(num/den)

    for i, (input, target) in enumerate(valid_loader):
        targets.append(target.numpy())

        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target.cuda(), volatile=True)

        # compute output
        output = model(input_var)

        outputs.append(output.data.cpu().numpy())

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs) >= 0.5
    loss = criterion(outputs, targets)
    return loss


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 3000))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
