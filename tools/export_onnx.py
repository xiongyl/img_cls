import sys
sys.path.insert(0, 'img_cls')

import os
import time
import yaml
import json
import onnx
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision

from models import model_zoo
from utils import load_ckpt, mkdir_if_no_exist

model_names = list(model_zoo.keys())

parser = argparse.ArgumentParser(description='Export Classification ONNX Model')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--config', default='config.yaml')
parser.add_argument('--image-size', default=128, type=int,
                    help='image size (default: 128x128)')
parser.add_argument('--input-size', default=112, type=int,
                    help='input size (default: 112x112)')
parser.add_argument('--feature-dim', default=256, type=int,
                    metavar='D', help='feature dimension (default: 256)')
parser.add_argument('--num-classes', default=3, type=int,
                    metavar='N', help='number of classes (default: 3)')
parser.add_argument('--model-path', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--onnx-path', default='mask-cls.onnx', type=str,
                    help='path to store checkpoint (default: checkpoints)')



def main():
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config['common'].items():
        setattr(args, k, v)

    if not (args.model_path and os.path.isfile(args.model_path)):
        print("=> no checkpoint found at '{}'".format(args.model_path))
        return

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = model_zoo[args.arch](num_classes=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()

    load_ckpt(args.model_path, model)

    torch_in = torch.rand(1, 3, args.input_size, args.input_size)
    torch_in = torch_in.cuda()

    model = model.module
    model.eval()
    #print(model)
    torch_out = torch.onnx.export(model, torch_in, args.onnx_path, verbose=True, opset_version=7)
    print("Exporting onnx model to {}".format(args.onnx_path))

    transforms = []
    transforms.append({'resize': args.image_size})
    transforms.append({'center_crop': args.input_size})
    transforms.append({'to_tensor': None})
    transforms.append({'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std':[0.229, 0.224, 0.225]
                    }})
    config = {'transforms': transforms}

    onnx_config_path = args.onnx_path.replace('.onnx', '-cfg.json')
    print("Exporting onnx model config to {}".format(onnx_config_path))
    with open(onnx_config_path , 'w') as f:
        json.dump(config, f)

if __name__ == '__main__':
    main()
