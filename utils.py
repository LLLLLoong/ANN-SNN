import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from modules import TCL, MyFloor, ScaledNeuron, StraightThrough, MyFloor_Channel, MyFloor_Layer
from modules import MPLayer
import math

def isActivation(name):
    if 'relu' in name.lower() or 'clip' in name.lower() or 'floor' in name.lower() or 'tcl' in name.lower():
        return True
    return False

def isLayerActivation(name):
    if 'myfloor_layer' in name.lower():
        return True
    return False

def replace_MPLayer_by_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_MPLayer_by_neuron(module)
        if module.__class__.__name__ == 'MPLayer':
            model._modules[name] = ScaledNeuron(scale=module.v_threshold)
    return model


def replace_activation_by_MPLayer(model, presim_len, sim_len):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_MPLayer(module, presim_len, sim_len)
        if isActivation(module.__class__.__name__.lower()):
            if module.mode == 'exp':
                model._modules[name] = MPLayer(v_threshold=torch.exp(module.up), presim_len=presim_len, sim_len=sim_len)
            elif module.mode == 'log':
                model._modules[name] = MPLayer(v_threshold=F.softplus(module.up), presim_len=presim_len, sim_len=sim_len)
            elif module.mode == 'log_modified':
                model._modules[name] = MPLayer(v_threshold=(F.softplus(module.up)-module.up), presim_len=presim_len, sim_len=sim_len)
            else:
                model._modules[name] = MPLayer(v_threshold=module.up, presim_len=presim_len, sim_len=sim_len)
    return model

def replace_activation_by_module(model, m):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_module(module, m)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "up"):
                print(module.up.item())
                model._modules[name] = m(module.up.item())
            else:
                model._modules[name] = m()
    return model

def replace_activation_by_floor(model, t):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_floor(module, t)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "up"):
                print(module.up.item())
                if t == 0:
                    model._modules[name] = TCL()
                else:
                    model._modules[name] = MyFloor(module.up.item(), t)
            else:
                if t == 0:
                    model._modules[name] = TCL()
                else:
                    model._modules[name] = MyFloor(8., t)
    return model

# 全局变量，记录当前更改的是第几个激活函数
global_activation_num = 0
total_activation_num_dict = {'resnet18':17, 'resnet34':33, 'resnet20':19, 'vgg16':15, 'resnet50':49, 'resnet101':100}
def replace_activation_by_floor_mix(model, t, mode = 'softplus', channel_num = 3, model_name='resnet34', init_threshold=8.):
    print('model name: ', model_name)
    print('activation mode: ', mode)
    total_activation_num = total_activation_num_dict[model_name]
    up_init = init_threshold
    
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_floor_mix(module, t, mode, channel_num, model_name, init_threshold)
        if isActivation(module.__class__.__name__.lower()):
            global global_activation_num
            global_activation_num += 1
            print('global_activation_num: {}, total_activation_num: {}, channel_num: {}'.format(global_activation_num, total_activation_num, channel_num))
            if hasattr(module, "up"):
                # print(module.up.item())
                if t == 0:
                    model._modules[name] = TCL()
                else:
                    if global_activation_num > total_activation_num - channel_num:
                        model._modules[name] = MyFloor_Channel(module.up.item(), t, mode)
                    else:
                        model._modules[name] = MyFloor_Layer(module.up.item(), t, mode)
            else:
                if t == 0:
                    model._modules[name] = TCL()
                else:
                    if global_activation_num > total_activation_num - channel_num:
                        model._modules[name] = MyFloor_Channel(up_init, t, mode)
                    else:
                        model._modules[name] = MyFloor_Layer(up_init, t, mode)
    return model

global_activation_num2 = 0
# 替换层级别激活函数为通道级别激活函数, 并使用层级别的激活函数值来初始化通道级别的激活函数阈值
def replace_layer_activation_by_channel(model, t, mode = 'softplus', channel_num = 3, model_name='resnet34'):
    total_activation_num = total_activation_num_dict[model_name]
    
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_layer_activation_by_channel(module, t, mode, channel_num, model_name)
        if isLayerActivation(module.__class__.__name__.lower()):
            global global_activation_num2
            global_activation_num2 += 1
            print('global_activation_num: {}, total_activation_num: {}, channel_num: {}'.format(global_activation_num2, total_activation_num, channel_num))
            if hasattr(module, "up"):
                print(module.up.item())
                if t == 0:
                    model._modules[name] = TCL()
                else:
                    if global_activation_num2 > total_activation_num - channel_num:
                        model._modules[name] = MyFloor_Channel(module.up.item(), t, mode)
                    else:
                        model._modules[name] = MyFloor_Layer(module.up.item(), t, mode)
    return model

def replace_activation_by_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_activation_by_neuron(module)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "up"):
                if module.mode == 'softplus':
                    model._modules[name] = ScaledNeuron(scale=F.softplus(module.up))
                else:
                    model._modules[name] = ScaledNeuron(scale=module.up)
            else:
                model._modules[name] = ScaledNeuron(scale=1.)
    return model

def replace_maxpool2d_by_avgpool2d(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model

def reset_net(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            reset_net(module)
        if 'Neuron' in module.__class__.__name__:
            module.reset()
    return model

def _fold_bn(conv_module, bn_module, avg=False):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module, avg=False):
    w, b = _fold_bn(conv_module, bn_module, avg)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2

def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)


def search_fold_and_remove_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if is_bn(m) and is_absorbing(prev):
            fold_bn_into_conv(prev, m)
            # set the bn module to straight through
            setattr(model, n, StraightThrough())
        elif is_absorbing(m):
            prev = m
        else:
            prev = search_fold_and_remove_bn(m)
    return prev


def regular_set(model, paras=([],[],[])):
    for n, module in model._modules.items():
        if isActivation(module.__class__.__name__.lower()) and hasattr(module, "up"):
            for name, para in module.named_parameters():
                paras[0].append(para)
        elif 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                paras[2].append(para)
        elif len(list(module.children())) > 0:
            paras = regular_set(module, paras)
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                paras[1].append(para)
    return paras

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()