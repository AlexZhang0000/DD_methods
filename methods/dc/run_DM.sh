# DM: IPC=1
python main_DM.py --dataset CIFAR10 --model ConvNet --ipc 1 --init real --lr_img 1 \
  --dsa_strategy color_crop_cutout_flip_scale_rotate --save_path distilled_results/DM/CIFAR10/ipc1

# DM: IPC=10
python main_DM.py --dataset CIFAR10 --model ConvNet --ipc 10 --init real --lr_img 1 \
  --dsa_strategy color_crop_cutout_flip_scale_rotate --save_path distilled_results/DM/CIFAR10/ipc10

# DM: IPC=50
python main_DM.py --dataset CIFAR10 --model ConvNet --ipc 50 --init real --lr_img 1 \
  --dsa_strategy color_crop_cutout_flip_scale_rotate --save_path distilled_results/DM/CIFAR10/ipc50
