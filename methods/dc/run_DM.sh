#!/bin/bash

# 定义参数
DATASET="CIFAR10"
MODEL="ConvNet"
EVAL_MODE="SS"
ITER=20000
NUM_EXP=1
NUM_EVAL=5
EPOCH_EVAL_TRAIN=1000
LR_IMG=1.0
LR_NET=0.01
BATCH_REAL=256
BATCH_TRAIN=256
INIT="real"
DSA_STRATEGY="color_crop_cutout_flip_scale_rotate"
DATA_PATH="data"
SAVE_PATH="result"
DIS_METRIC="ours"

# IPC 列表
for IPC in 1 10 50
do
  echo "================= Generating CIFAR10 | IPC=${IPC} ================="
  python main_DM.py \
    --dataset ${DATASET} \
    --model ${MODEL} \
    --ipc ${IPC} \
    --eval_mode ${EVAL_MODE} \
    --Iteration ${ITER} \
    --num_exp ${NUM_EXP} \
    --num_eval ${NUM_EVAL} \
    --epoch_eval_train ${EPOCH_EVAL_TRAIN} \
    --lr_img ${LR_IMG} \
    --lr_net ${LR_NET} \
    --batch_real ${BATCH_REAL} \
    --batch_train ${BATCH_TRAIN} \
    --init ${INIT} \
    --dsa_strategy ${DSA_STRATEGY} \
    --data_path ${DATA_PATH} \
    --save_path ${SAVE_PATH} \
    --dis_metric ${DIS_METRIC}
done
