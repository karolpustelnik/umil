#!/bin/bash
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=40
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash tools/dist_umil_recognizer.sh 4
#bash tools/dist_train_recognizer.sh 4

