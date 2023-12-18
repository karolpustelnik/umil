#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
$PYTHON -m torch.distributed.launch --nproc_per_node=$1 main_umil.py -cfg configs/chad/32_5.yaml --output output/chad8_lower_lr --pretrained k400_16_8.pth \