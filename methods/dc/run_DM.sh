#!/bin/bash

echo "🟩 Installing dependencies..."
pip install -r requirements.txt

for IPC in 1 10 50
do
  echo "🔁 Synthesizing CIFAR10 with DM | IPC=$IPC..."
  python main.py \
    --dataset CIFAR10 \
    --model ConvNet \
    --ipc $IPC \
    --init real \
    --dsa_strategy color_crop_cutout_flip_scale_rotate \
    --lr_img 1 \
    --num_exp 1 \
    --num_eval 1
done

for IPC in 1 10 50
do
  echo "🧪 Evaluating synthetic IPC=$IPC using model trained on real CIFAR10..."
  python train.py \
    --train_dataset=./data/CIFAR10 \
    --test_dataset=./result/res_DM_CIFAR10_ConvNet_${IPC}ipc.pt \
    --model=ConvNet \
    --mode=test_syn \
    --epochs=100
done

for IPC in 1 10 50
do
  echo "🧪 Training on synthetic IPC=$IPC, evaluating on real CIFAR10..."
  python train.py \
    --train_dataset=./result/res_DM_CIFAR10_ConvNet_${IPC}ipc.pt \
    --test_dataset=./data/CIFAR10 \
    --model=ConvNet \
    --mode=test_real \
    --epochs=300
done

echo "✅ All steps finished."
