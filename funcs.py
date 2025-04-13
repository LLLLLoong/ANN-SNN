import numpy as np
from torch import nn
import torch
from tqdm import tqdm
from utils import *
from modules import LabelSmoothing
import torch.distributed as dist
import random
import os
from warmup_scheduler import GradualWarmupScheduler
import torch.nn.functional as F

def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=False

def mp_test(test_dataloader, model, net_arch, presim_len, sim_len, device):
    new_tot = torch.zeros(sim_len).cuda(device)
    model = model.cuda(device)
    model.eval()
    
    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            new_spikes = 0
            img = img.cuda(device)
            label = label.cuda(device)
            
            for t in range(presim_len+sim_len):
                out = model(img)
                
                if t >= presim_len:
                    new_spikes += out
                    new_tot[t-presim_len] += (label==new_spikes.max(1)[1]).sum().item()
                   
    return new_tot


def eval_ann(test_dataloader, model, loss_fn, device, rank=0):
    epoch_loss = 0
    tot = torch.tensor(0.).cuda(device)
    model.eval()
    model.cuda(device)
    length = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            img = img.cuda(device)
            label = label.cuda(device)
            out = model(img)
            loss = loss_fn(out, label)
            epoch_loss += loss.item()
            length += len(label)    
            tot += (label==out.max(1)[1]).sum().data
    return tot/length, epoch_loss/length

def train_ann(train_dataloader, test_dataloader, model, epochs, device, loss_fn, lr=0.1, wd=5e-4, save=None, parallel=False, rank=0, dataset='cifar100',lr_scheduler='MuliStepLR', train_stage='train', activation_mode='origin'):
    model.cuda(device)
    para1, para2, para3 = regular_set(model)
    # para1 是 up 值
    optimizer = torch.optim.SGD([
                                {'params': para1, 'weight_decay': wd}, 
                                {'params': para2, 'weight_decay': wd}, 
                                {'params': para3, 'weight_decay': wd}
                                ],
                                lr=lr, 
                                momentum=0.9)
    
    if lr_scheduler=='MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
    elif lr_scheduler=='CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif lr_scheduler=='WarmupCosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.WarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=epochs)

    # first_iter在训练channel阈值时打开
    first_iter = False
    if train_stage=='ft':
        first_iter=True
    best_acc = 0
    for epoch in range(epochs):
        epoch_loss = 0
        length = 0
        model.train()
        # 每1分钟更新一次
        for img, label in tqdm(train_dataloader, mininterval=1):
            img = img.cuda(device)
            label = label.cuda(device)
            optimizer.zero_grad()
            out = model(img)
            # 更新optimizer的参数
            if first_iter:
                para1, para2, para3 = regular_set(model,([],[],[]))
                optimizer = torch.optim.SGD([
                                            {'params': para1, 'weight_decay': wd},
                                            {'params': para2, 'weight_decay': wd}, 
                                            {'params': para3, 'weight_decay': wd}
                                            ],
                                            lr=lr, 
                                            momentum=0.9)
                
                if lr_scheduler=='MultiStepLR':
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
                elif lr_scheduler=='CosineAnnealingLR':
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                elif lr_scheduler=='WarmupCosineAnnealingLR':
                    scheduler = torch.optim.lr_scheduler.WarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=epochs)
                first_iter = False

            loss = loss_fn(out, label)
            
            # 训练Softplus时对 log(e^lambda+1) 计算l2正则化
            if activation_mode=='softplus':
                lambda_l2 = 0.0005
            else:
                lambda_l2 = 0
            l2_regularization = torch.tensor(0.).cuda(device)
            for para in para1:
                l2_regularization += F.softplus(para).sum()
            loss += lambda_l2 * l2_regularization            
            if loss.cpu().detach().numpy() > 1e8 or np.isnan(loss.cpu().detach().numpy()):
                raise ValueError('Training diverged...')
           
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if epoch_loss > 1e8 or np.isnan(epoch_loss):
                raise ValueError('Training diverged...')
            length += len(label)
        tmp_acc, val_loss = eval_ann(test_dataloader, model, loss_fn, device, rank)
        if parallel:
            dist.all_reduce(tmp_acc)
        print('Epoch {} -> Val_loss: {}, Acc: {}'.format(epoch, val_loss, tmp_acc), flush=True)
        if rank == 0 and save != None and tmp_acc >= best_acc:
            a = model.state_dict()
            torch.save(model.state_dict(), f'./saved_models/' + save + '.pth')
        best_acc = max(tmp_acc, best_acc)
        print('best_acc: ', best_acc)
        # scheduler.step()
        if lr_scheduler=='ReduceLROnPlateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
    return best_acc, model

def eval_snn(test_dataloader, model, device, sim_len=8, rank=0):
    tot = torch.zeros(sim_len).cuda(device)
    length = 0
    model = model.cuda(device)
    model.eval()
    # valuate
    with torch.no_grad():
        for idx, (img, label) in enumerate(tqdm(test_dataloader)):
            spikes = 0
            length += len(label)
            img = img.cuda()
            label = label.cuda()
            for t in range(sim_len):
                out = model(img)
                spikes += out
                tot[t] += (label==spikes.max(1)[1]).sum()
            reset_net(model)
    return tot/length