#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
$PYTHON -m torch.distributed.launch --nproc_per_node=$1 main_umil.py -cfg configs/chad/32_5.yaml --output output/test --pretrained /workspace/umil/output/chad8/ckpt_epoch_4.pth --only_test
