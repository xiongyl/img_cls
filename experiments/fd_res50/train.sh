#!/bin/bash
work_path=$(dirname $0)

num=8
GPU=0,1,2,3,4,5,6,7
#num=2
#GPU=0,1

CUDA_VISIBLE_DEVICES=$GPU mpirun -np $num \
python tools/train.py \
        --distributed \
        --pretrained 'pretrained_models/resnet50-19c8e357.pth' \
        --config $work_path/config.yaml
