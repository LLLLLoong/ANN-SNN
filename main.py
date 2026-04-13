import math
import torch.multiprocessing as mp
import argparse
from Models import modelpool
from Preprocess import datapool
from funcs import *
from utils import replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # debug snn
    parser.add_argument('--action', default='train', type=str, help='Action: train or test.')
    parser.add_argument('--gpus', default=1, type=int, help='GPU number to use.')
    parser.add_argument('--bs', default=128, type=int, help='Batchsize')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate') 
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=2, type=int, help='Training epochs') # better if set to 300 for CIFAR dataset
    parser.add_argument('--id', default='train_test', type=str, help='Model identifier')
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--l', default=4, type=int, help='L')
    parser.add_argument('--t', default=64, type=int, help='T')
    parser.add_argument('--mode', type=str, default='ann')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--activation_mode', type=str, default='origin')
    parser.add_argument('--channel_num', type=int, default=0)
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR')
    parser.add_argument('--init_threshold', type=float, default=4.)
    parser.add_argument('--presim_len', type=int, default=0, help='Pre Simulation length of COS')
    args = parser.parse_args()
    print(args)
    seed_all(args.seed)

    # only ImageNet using multiprocessing,
    if args.gpus > 1:
        if args.data.lower() != 'imagenet':
            AssertionError('Only ImageNet using multiprocessing.')
    else:
        # preparing data
        train, test = datapool(args.data, args.bs)
        # preparing model
        model = modelpool(args.model, args.data)
        model = replace_maxpool2d_by_avgpool2d(model)
        # 替换激活函数
        model = replace_activation_by_floor_mix(model, t=args.l, mode=args.activation_mode, channel_num=args.channel_num, model_name = args.model, init_threshold=args.init_threshold)
        print(model)
        criterion = nn.CrossEntropyLoss()
        if args.action == 'train':
            train_ann(train, test, model, args.epochs, args.device, criterion, args.lr, args.wd, model_name=args.model, dataset=args.data, lr_scheduler=args.lr_scheduler, train_stage='train', activation_mode=args.activation_mode, L=args.l)
        elif args.action == 'test' or args.action == 'evaluate':
            model.cuda(args.device)
            model.eval()
            for img, label in train:
                img = img.cuda(args.device)
                out = model(img)
                break

            save_dir = f'./saved_models/{args.data}/{args.model}'
            save_name = f'{args.activation_mode}_T[{args.l}]'
            model.load_state_dict(torch.load(f'{save_dir}/{save_name}.pth'))
            
            if args.presim_len > 0:
                cap_dataset = 10000
                replace_activation_by_MPLayer(model,presim_len=args.presim_len,sim_len=args.t)
                if args.presim_len > 0:
                    new_acc = mp_test(test, model, net_arch=args.model, presim_len=args.presim_len, sim_len=args.t, device=args.device)
                else:         
                    replace_MPLayer_by_neuron(model)
                    new_acc = eval_snn(test, model, sim_len=args.t, device=args.device)

                t = 1
                while t < args.t:
                    print(f'time step {t}, Accuracy = {(new_acc[t-1] / cap_dataset):.4f}')
                    t *= 2
                print(f'time step {args.t}, Accuracy = {(new_acc[args.t-1] / cap_dataset):.4f}')
            else:
                if args.mode == 'snn':
                    print('eval snn')
                    model = replace_activation_by_neuron(model)
                    model.to(args.device)
                    acc = eval_snn(test, model, args.device, args.t)
                    print('Accuracy: ', acc)
                    # 打印不同的T的acc
                    for i in range(0,int(math.log2(args.t)+1)):
                        print(f'Accuracy of T={2**i}', acc[2**i-1])
                elif args.mode == 'ann':
                    model.to(args.device)
                    acc, _ = eval_ann(test, model, criterion, args.device)
                    print('Accuracy: {:.4f}'.format(acc))
                else:
                    AssertionError('Unrecognized mode')
