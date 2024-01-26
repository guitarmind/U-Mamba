#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python /kaggle/input/u-mamba/inference.py \
    --model-folder $1 \
    --output-path model_1_pred.pkl \
    --device 0 \
    --checkpoint best &
CUDA_VISIBLE_DEVICES=1 python /kaggle/input/u-mamba/inference.py \
    --model-folder $2 \
    --output-path model_2_pred.pkl \
    --device 0 \
    --checkpoint best &
wait