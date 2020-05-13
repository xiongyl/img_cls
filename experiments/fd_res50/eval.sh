#!/bin/bash
work_path=$(dirname $0)

num=1
GPU=0

CUDA_VISIBLE_DEVICES=$GPU mpirun -np $num \
python tools/eval.py \
        --distributed \
        --evaluate \
        --model-path 'output/malong/checkpoints/model_best.pth.tar' \
        --eval-root fd_data/ \
        --eval-list fd_data/annotations/aizoo_crop_val.txt \
        --config $work_path/config.yaml
