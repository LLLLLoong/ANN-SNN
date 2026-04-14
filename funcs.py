import numpy as np
import time
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

def train_ann(train_dataloader, test_dataloader, model, epochs, device, loss_fn, lr=0.1, wd=5e-4, model_name='resnet18', parallel=False, rank=0, dataset='cifar100',lr_scheduler='MuliStepLR', train_stage='train', activation_mode='origin', L=4, resume=False):
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

    save_dir = f'./saved_models/{dataset}/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    save_name = f'{activation_mode}_T[{L}]'
    log_file = os.path.join(save_dir, f'{save_name}_log.txt')
    
    ckpt_dir = f'./saved_models/checkpoints/{dataset}/{model_name}'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'ckpt_{activation_mode}_T[{L}]_{dataset}_{model_name}_latest.pth')

    start_epoch = 0
    best_acc = 0

    if resume and os.path.exists(ckpt_path):
        if rank == 0:
            print(f"==> Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if rank == 0:
            print(f"==> Loaded checkpoint from epoch {checkpoint['epoch']} with best_acc: {best_acc:.4f}\n")

    if rank == 0:
        mode = 'a' if resume and os.path.exists(ckpt_path) else 'w'
        with open(log_file, mode) as f:
            if mode == 'w':
                f.write("Epoch\tTrain_Loss\tVal_Loss\tVal_Acc\tTime(s)\n")

    # first_iter在训练channel阈值时打开
    first_iter = False
    if train_stage=='ft':
        first_iter=True

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
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
        train_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        epoch_time = time.time() - start_time
        print('Epoch {} -> Train_loss: {:.4f}, Val_loss: {:.4f}, Acc: {:.4f}, Time: {:.2f}s'.format(epoch, train_loss, val_loss, tmp_acc, epoch_time), flush=True)

        if rank == 0:
            with open(log_file, 'a') as f:
                f.write(f"{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{tmp_acc:.4f}\t{epoch_time:.2f}\n")

            if tmp_acc >= best_acc:
                torch.save(model.state_dict(), os.path.join(save_dir, f'{save_name}.pth'))
                
            # Save checkpoint for resume
            ckpt_state = {
                'epoch': epoch,
                'best_acc': max(tmp_acc, best_acc),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'args': {'activation_mode': activation_mode, 'L': L, 'dataset': dataset, 'model_name': model_name}
            }
            torch.save(ckpt_state, ckpt_path)

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