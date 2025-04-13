import math
import torch.multiprocessing as mp
import argparse
from Models import modelpool
from Preprocess import datapool
from funcs import *
from utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d, replace_layer_activation_by_channel
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--action', default='train', type=str, help='Action: train or test.')
    parser.add_argument('--gpus', default=1, type=int, help='GPU number to use.')
    parser.add_argument('--bs', default=128, type=int, help='Batchsize')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate') 
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=100, type=int, help='Finetune epochs') # better if set to 300 for CIFAR dataset
    # 层级别模型的id
    parser.add_argument('--layer_id', default='train_test', type=str, help='Model identifier')
    # 通道级别模型的id
    parser.add_argument('--id', default='ft_test', type=str, help='Model identifier')
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--l', default=8, type=int, help='L')
    parser.add_argument('--t', default=64, type=int, help='T')
    parser.add_argument('--mode', type=str, default='ann')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--activation_mode', type=str, default='softplus')
    parser.add_argument('--channel_num', type=int, default=17)
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR')
    # 使用layer级别的threshold作为channel的初始化阈值
    parser.add_argument('--init_threshold', type=float, default=4.)
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
        
        # 1. 先构建层级别的模型
        # 2. 加载层级别权重
        # 3. 替换激活函数为通道级别激活函数
        
        # 层级别的channel_num = 0
        model = replace_activation_by_floor_mix(model, t=args.l, mode=args.activation_mode, channel_num=0, model_name = args.model, init_threshold=args.init_threshold)
        # 加载权重
        model.load_state_dict(torch.load('./saved_models/'+ args.layer_id + '.pth'))
        # 替换激活函数为通道级别
        model = replace_layer_activation_by_channel(model, t=args.l, mode=args.activation_mode, channel_num=args.channel_num, model_name = args.model)
        
        print(model)
        criterion = nn.CrossEntropyLoss()
        if args.action == 'train':
            train_ann(train, test, model, args.epochs, args.device, criterion, args.lr, args.wd, args.id, dataset=args.data, lr_scheduler=args.lr_scheduler, train_stage='ft', activation_mode=args.activation_mode)
