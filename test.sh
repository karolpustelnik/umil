#!/bin/bash

export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=1

bash tools/dist_test_recognizer.sh 1
