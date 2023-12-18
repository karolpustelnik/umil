#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$1 main.py -cfg configs/chad/32_5.yaml --batch-size 1 --accumulation-steps 16 --output output/mil --pretrained k400_16_8.pth