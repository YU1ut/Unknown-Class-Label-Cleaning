from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

import models.resnet as models
import dataset.cifar10 as dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Optimization options
parser.add_argument('--pre-epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run for label cleaning stage')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run for fine-tune stage')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--pre-lr', default=0.03, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[40, 80],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--out', default='result',
                    help='Directory to output the result')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--percent', type=float, default=0.4,
                    help='Percentage of noise')
parser.add_argument('--clean', action='store_true',
                    help='Whether use the dataset with only clean labels.')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='Hyper parameter alpha of loss function')
parser.add_argument('--beta', type=float, default=0.5,
                    help='Hyper parameter beta of loss function')
parser.add_argument('--lam', type=float, default=0.4,
                    help='The proportion of the open-set noisy labels in the loss function')
parser.add_argument('--begin', type=int, default=20,
                    help='When to begin updating labels')
parser.add_argument('--ood', default='cifar100',
                    help='The ')
parser.add_argument('--close', action='store_true',
                    help='Whether use add closed-set noise to the dataset.')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing open-set nosiy cifar10')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = dataset.get_cifar10('./data', args, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10('./data', train=False, transform=transform_test)

    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print("==> creating resnet34")
    model = models.resnet34(num_classes=11)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    val_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.pre_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'noisy-cifar-10'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    # Label update stage
    for epoch in range(start_epoch, args.pre_epochs):
        # adjust_learning_rate(optimizer, epoch + 1)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['pre_lr']))

        train_loss, train_acc = train(trainloader, model, optimizer, epoch, use_cuda)
        val_loss, val_acc = validate(testloader, model, val_criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['pre_lr'], train_loss, val_loss, train_acc, val_acc])

        # save model
        is_best = val_acc > best_acc

        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': val_acc,
                'best_val_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # Fine-tune stage
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch + 1)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train_soft(trainloader, model, optimizer, epoch, use_cuda)
        val_loss, val_acc = validate(testloader, model, val_criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, val_loss, train_acc, val_acc])

        # save model
        is_best = val_acc > best_acc

        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': val_acc,
                'best_val_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        

    logger.close()
    logger.plot()
    savefig(os.path.join(args.out, 'log.eps'))

    print(f'Best val acc: {best_acc}')

def train(trainloader, model, optimizer, epoch, use_cuda):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    results = np.zeros((len(trainloader.dataset), 11), dtype=np.float32)
    pred = np.zeros((len(trainloader.dataset)), dtype=np.float32)
    gt = np.zeros((len(trainloader.dataset)), dtype=np.float32)

    bar = Bar('Training', max=len(trainloader))
    for batch_idx, (inputs, targets, soft_targets, indexs) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            soft_targets= soft_targets.cuda(non_blocking=True)

        # compute output
        outputs = model(inputs)
        
        probs, loss = mycriterion(outputs, soft_targets)

        results[indexs.numpy().tolist()] = probs.cpu().detach().numpy().tolist()
        pred[indexs.numpy().tolist()] = probs.argmax(1).cpu().detach().numpy().tolist()
        gt[indexs.numpy().tolist()] = targets.cpu().detach().numpy().tolist()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()

    cal_acc(gt, pred)

    trainloader.dataset.label_update(results)

    return (losses.avg, top1.avg)

def train_soft(trainloader, model, optimizer, epoch, use_cuda):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    pred = np.zeros((len(trainloader.dataset)), dtype=np.float32)
    gt = np.zeros((len(trainloader.dataset)), dtype=np.float32)


    bar = Bar('Training', max=len(trainloader))
    for batch_idx, (inputs, targets, soft_targets, indexs) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            soft_targets= soft_targets.cuda(non_blocking=True)

        # compute output
        outputs = model(inputs)
        
        probs = torch.softmax(outputs, dim=1)

        loss = mycriterion_soft(outputs, soft_targets)

        pred[indexs.numpy().tolist()] = probs.argmax(1).cpu().detach().numpy().tolist()
        gt[indexs.numpy().tolist()] = targets.cpu().detach().numpy().tolist()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()

    cal_acc(gt, pred)

    return (losses.avg, top1.avg)


def validate(valloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Testing ', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs[:,:-1], targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def cal_acc(gt_list, predict_list, num=11):
    acc_sum = 0
    y_ = []
    pred_y_ = []
    for n in range(num):
        y = []
        pred_y = []
        for i in range(len(gt_list)):
            gt = gt_list[i]
            predict = predict_list[i]
            if gt == n:
                y.append(gt)
                pred_y.append(predict)
        print ('{}: {:4f}'.format(n if n != (num - 1) else 'Unk', accuracy_score(y, pred_y)))
        if n == (num - 1):
            print ('Known Avg Acc: {:4f}'.format(acc_sum / (num - 1)))
        acc_sum += accuracy_score(y, pred_y)
    for i in range(len(gt_list)):
        gt = gt_list[i]
        predict = predict_list[i]
        if gt < num - 1:
            y_.append(gt)
            pred_y_.append(predict)
    print ('Known Acc: {:4f}'.format(accuracy_score(y_, pred_y_)))
    print ('Avg Acc: {:4f}'.format(acc_sum / num))
    print ('Overall Acc : {:4f}'.format(accuracy_score(gt_list, predict_list)))

def mycriterion(outputs, soft_targets):
    unk_prob = args.lam
    kn_prob = (1-unk_prob)/10
    p = torch.tensor([kn_prob, kn_prob, kn_prob, kn_prob, kn_prob, kn_prob, kn_prob, kn_prob, kn_prob, kn_prob, unk_prob]).cuda()

    probs = F.softmax(outputs, dim=1)
    avg_probs = torch.mean(probs, dim=0)

    L_c = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * soft_targets, dim=1))
    L_p = -torch.sum(torch.log(avg_probs) * p)
    L_e = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * probs, dim=1))

    loss = L_c + args.alpha * L_p + args.beta * L_e

    return probs, loss

def mycriterion_soft(outputs, soft_targets):
    probs = F.softmax(outputs, dim=1)
    loss = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * soft_targets, dim=1))

    return loss

if __name__ == '__main__':
    main()