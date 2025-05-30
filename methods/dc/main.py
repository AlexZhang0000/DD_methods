import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import wandb
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode')
    parser.add_argument('--num_exp', type=int, default=5, help='number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='number of evaluations per experiment')
    parser.add_argument('--epoch_eval_train', type=int, default=300)
    parser.add_argument('--Iteration', type=int, default=1000)
    parser.add_argument('--lr_img', type=float, default=0.1)
    parser.add_argument('--lr_net', type=float, default=0.01)
    parser.add_argument('--batch_real', type=int, default=256)
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--init', type=str, default='noise')
    parser.add_argument('--dsa_strategy', type=str, default='None')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--save_path', type=str, default='result')
    parser.add_argument('--dis_metric', type=str, default='ours')

    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    wandb.init(project="dataset-distillation", name=f"DM_{args.dataset}_{args.model}_{args.ipc}ipc", config=vars(args))

    if not os.path.exists(args.data_path): os.mkdir(args.data_path)
    if not os.path.exists(args.save_path): os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist() if args.eval_mode in ['S', 'SS'] else [args.Iteration]
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = {key: [] for key in model_eval_pool}
    data_save = []

    for exp in range(args.num_exp):
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        indices_class = [[] for _ in range(num_classes)]
        for i, lab in enumerate(labels_all): indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        def get_images(c, n):
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)
        if args.init == 'real':
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data

        optimizer_img = torch.optim.SGD([image_syn,], lr=args.lr_img, momentum=0.5)
        criterion = nn.CrossEntropyLoss().to(args.device)

        for it in range(args.Iteration+1):
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
                        _, _, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)

                    mean_acc = np.mean(accs)
                    std_acc = np.std(accs)
                    wandb.log({f"acc_mean_{model_eval}": mean_acc, f"acc_std_{model_eval}": std_acc, "iteration": it})

                    if it == args.Iteration:
                        accs_all_exps[model_eval] += accs

                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]*std[ch] + mean[ch]
                image_syn_vis.clamp_(0, 1)
                wandb.log({"syn_images": [wandb.Image(image_syn_vis, caption=f"IPC {args.ipc} Iter {it}")]})

            # training step (as in original)
            loss = torch.tensor(0.0).to(args.device)
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)

            for ol in range(args.outer_loop):
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = [_.detach().clone() for _ in gw_real]

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss += match_loss(gw_syn, gw_real, args)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()

            if it == args.Iteration:
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                save_name = os.path.join(args.save_path, f'res_{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc.pt')
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps}, save_name)

    for key in accs_all_exps:
        accs = accs_all_exps[key]
        print(f'{key}: mean={np.mean(accs)*100:.2f}%, std={np.std(accs)*100:.2f}%')
        wandb.log({f"final_mean_acc_{key}": np.mean(accs), f"final_std_acc_{key}": np.std(accs)})

if __name__ == '__main__':
    main()


