#!/bin/bash

# ---------- 环境配置 ----------
ENV_NAME=distillation
ENV_YAML=requirements_11_3.yaml

echo "🟩 Step 0: 创建 Conda 环境 [$ENV_NAME] ..."
conda env create -f $ENV_YAML -n $ENV_NAME || { echo "⚠️ 环境可能已存在，跳过创建"; }
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME || { echo "❌ 激活环境失败"; exit 1; }

# ---------- 目录设置 ----------
DATASET="CIFAR10"
MODEL="ConvNet"
DATA_PATH="./data"
BUFFER_PATH="./buffer"

# ---------- Step 1: 生成 expert trajectories ----------
echo "🟦 Step 1: Generating expert trajectories ..."
python buffer.py \
  --dataset=$DATASET \
  --model=$MODEL \
  --train_epochs=50 \
  --num_experts=100 \
  --zca \
  --buffer_path=$BUFFER_PATH \
  --data_path=$DATA_PATH

# ---------- Step 2-5: 蒸馏 + 训练 + 测试 ----------
for IPC in 1 10 50
do
  echo "🟨 Step 2: Distilling CIFAR10 with IPC=$IPC ..."

  # 参数配置
  if [ "$IPC" == "1" ]; then
    PARAMS="--syn_steps=50 --expert_epochs=2 --max_start_epoch=2 --lr_img=100 --lr_lr=1e-07 --lr_teacher=0.01 --zca"
  elif [ "$IPC" == "10" ]; then
    PARAMS="--syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.001 --zca"
  elif [ "$IPC" == "50" ]; then
    PARAMS="--syn_steps=30 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.001"
  fi

  # 执行蒸馏
  python distill.py \
    --dataset=$DATASET \
    --ipc=$IPC \
    $PARAMS \
    --buffer_path=$BUFFER_PATH \
    --data_path=$DATA_PATH

  # Step 3: 用真实数据训练模型 ➜ 测试合成数据
  echo "🟦 Step 3: Train on real, test on synthetic IPC=$IPC ..."
  python train.py \
    --train_dataset=${DATA_PATH}/${DATASET} \
    --test_dataset=./images_${IPC}/ \
    --model=$MODEL \
    --epochs=100 \
    --mode=test_syn

  # Step 4: 用合成数据训练模型 ➜ 测试真实数据
  echo "🟩 Step 4: Train on synthetic IPC=$IPC, test on real ..."
  python train.py \
    --train_dataset=./images_${IPC}/ \
    --test_dataset=${DATA_PATH}/${DATASET} \
    --model=$MODEL \
    --epochs=300 \
    --mode=test_real

done

echo "🎉 All processes completed successfully!"
