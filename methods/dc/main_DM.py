# 完整版 main_DM.py，不省略任何结构，仅添加 wandb 支持和修复目录创建
import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug

import wandb

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode')
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='initialize synthetic images from noise or real')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='DSA strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    args = parser.parse_args()
    args.method = 'DM'
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    os.makedirs(args.data_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    wandb.init(project="dataset-distillation", name=f"DM_{args.dataset}_{args.model}_{args.ipc}ipc", config=vars(args))

    eval_it_pool = np.arange(0, args.Iteration+1, 2000).tolist() if args.eval_mode in ['S', 'SS'] else [args.Iteration]
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = {key: [] for key in model_eval_pool}
    data_save = []

    for exp in range(args.num_exp):
        print(f'\n================== Exp {exp} ==================\n')
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        indices_class = [[] for _ in range(num_classes)]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        def get_images(c, n):
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, device=args.device).view(-1)

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach()
        else:
            print('initialize synthetic data from random noise')

        optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)

        for it in range(args.Iteration+1):
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                    acc_mean = np.mean(accs)
                    wandb.log({f"eval_acc_iter{it}_{model_eval}": acc_mean})
                    print(f'IPC {args.ipc} | Eval {model_eval} | Iter {it} | acc = {acc_mean:.4f}')
                    if it == args.Iteration:
                        accs_all_exps[model_eval] += accs

                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                image_syn_vis = torch.clamp(image_syn_vis, 0, 1)
                save_path = os.path.join(args.save_path, f'vis_{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc_exp{exp}_iter{it}.png')
                save_image(image_syn_vis, save_path, nrow=args.ipc)

            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.train()
            for param in net.parameters():
                param.requires_grad = False
            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed

            loss = torch.tensor(0.0).to(args.device)
            if 'BN' not in args.model:
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)
                    loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
            else:
                images_real_all, images_syn_all = [], []
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    images_real_all.append(img_real)
                    images_syn_all.append(img_syn)
                output_real = embed(torch.cat(images_real_all)).detach()
                output_syn = embed(torch.cat(images_syn_all))
                output_real = output_real.reshape(num_classes, args.batch_real, -1)
                output_syn = output_syn.reshape(num_classes, args.ipc, -1)
                loss += torch.sum((torch.mean(output_real, 1) - torch.mean(output_syn, 1))**2)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg = loss.item() / num_classes

            if it % 10 == 0:
                print(f'{get_time()} iter = {it:05d}, loss = {loss_avg:.4f}')
                wandb.log({"iteration": it, "loss": loss_avg})

            if it == args.Iteration:
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps},
                           os.path.join(args.save_path, f'res_{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc.pt'))

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print(f'Run {args.num_exp} experiments | Eval model {key} | Mean acc = {np.mean(accs)*100:.2f}% | Std = {np.std(accs)*100:.2f}%')

if __name__ == '__main__':
    main()



