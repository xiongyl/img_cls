import multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn')

import sys
sys.path.insert(0, 'img_cls')

import os
import time
import yaml
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from models import model_zoo
from datasets import FileListDatasetName
from utils import (AverageMeter, accuracy, save_ckpt, load_ckpt,
                    init_processes, mkdir_if_no_exist, get_cat2cls)
from logger import create_logger

from autoaugment import ImageNetPolicy, CIFAR10Policy

model_names = list(model_zoo.keys())

parser = argparse.ArgumentParser(description='Image Classification')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--eval-root', type=str)
parser.add_argument('--eval-list', nargs='+', type=str)
#parser.add_argument('--val-root', type=str)
#parser.add_argument('--val-list', nargs='+', type=str)
parser.add_argument('--config', default='config.yaml')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--batch-size', default=256, type=int,
                    metavar='M', help='mini-batch size')
parser.add_argument('--test-batch-size', default=256, type=int,
                    help='mini-batch size during test')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--image-size', default=128, type=int,
                    help='image size (default: 128x128)')
parser.add_argument('--input-size', default=112, type=int,
                    help='input size (default: 112x112)')
parser.add_argument('--feature-dim', default=256, type=int,
                    metavar='D', help='feature dimension (default: 256)')
parser.add_argument('--num-classes', default=3, type=int,
                    metavar='N', help='number of classes (default: 2)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--model-path', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-path', default='checkpoints/ckpt', type=str,
                    help='path to store checkpoint (default: checkpoints)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--distributed', dest='distributed', action='store_true',
                    help='distributed training')
parser.add_argument('--dist-addr', default='127.0.0.1', type=str,
                    help='distributed address')
parser.add_argument('--dist-port', default='23456', type=str,
                    help='distributed port')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--output-fn', default='validate.txt', type=str)

best_prec1 = 0


class ColorAugmentation(object):
    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec == None:
            eig_vec = torch.Tensor([
                [ 0.4009,  0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [ 0.4203, -0.6948, -0.5836],
            ])
        if eig_val == None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val))*0.1
        quatity = torch.mm(self.eig_val*alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


def main():
    global args, best_prec1
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config['common'].items():
        setattr(args, k, v)
    print(args.eval_list)

    if not (args.model_path and os.path.isfile(args.model_path)):
        print("=> no checkpoint found at '{}'".format(args.model_path))
        return

    gpu_num = torch.cuda.device_count()

    if args.distributed:
        args.rank, args.size = init_processes(args.dist_addr, args.dist_port, gpu_num, args.dist_backend)
        print("=> using {} GPUS for distributed training".format(args.size))
    else:
        args.rank = 0
        print("=> using {} GPUS for training".format(gpu_num))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = model_zoo[args.arch](num_classes=args.num_classes)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, [args.rank]) 
        print('create DistributedDataParallel model successfully', args.rank)

    if args.rank == 0:
        mkdir_if_no_exist(args.save_path, subdirs=['events/', 'logs/', 'checkpoints/'])
        logger = create_logger('global_logger', '{}/logs/log.txt'.format(args.save_path))
        logger.debug(args) # log args only to file
    else:
        logger = None

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    load_ckpt(args.model_path, model)

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        FileListDatasetName(
        args.eval_list,
        args.eval_root,
        transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.input_size),
            #transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate(val_loader, model, criterion, logger, args.print_freq, args.rank)

def validate(val_loader, model, criterion,
            logger, print_freq, rank, cls2tcls=None,
            save_path = '.'):
    n = len(val_loader)
    batch_time = AverageMeter(n)
    losses = AverageMeter(n)
    top1 = AverageMeter(n)
    if cls2tcls is not None:
        tcls_top1 = AverageMeter(n)

    model.eval()

    preds = []
    targets = []
    filenames = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target, filename) in enumerate(val_loader):
            filenames += filename
            target = target.cuda(non_blocking=True)

            output = model(input)
            #loss = criterion(output, target)
            #losses.update(loss.item())

            prec1, pred = accuracy(output, target, topk=(1,))
            top1.update(prec1[0][0].item())

            preds.append(pred.view(-1).cpu().data.numpy())
            targets.append(target.view(-1).cpu().data.numpy())

            if cls2tcls is not None:
                prec1, _ = accuracy(output, target, topk=(1,), cls2tcls=cls2tcls)
                tcls_top1.update(prec1[0][0].item())

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0 and rank == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      #'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, #loss=losses,
                       top1=top1))
        if rank == 0:
            logger.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
            if cls2tcls is not None:
                logger.info(' * Prec@1 {tcls_top1.avg: .3f}'.format(tcls_top1=tcls_top1))
            preds = np.hstack(preds)
            targets = np.hstack(targets)
            lines = ['{} {} {}\n'.format(fn, pred, target) for fn, pred, target in zip(filenames, preds, targets)]
            with open(args.output_fn, 'w') as f:
                f.writelines(lines)


if __name__ == '__main__':
    main()
