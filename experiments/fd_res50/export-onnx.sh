#!/bin/bash
work_path=$(dirname $0)

GPU=0

CUDA_VISIBLE_DEVICES=$GPU \
python tools/export_onnx.py \
        --config $work_path/config.yaml \
        --model-path output/malong/checkpoints/model_best.pth.tar \
        --onnx-path output/malong/checkpoints/mask-cls.onnx
