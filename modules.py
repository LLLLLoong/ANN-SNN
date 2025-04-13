from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Function
from spikingjelly.clock_driven import neuron

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

# SRP中的IFNeuron
class ScaledNeuron(nn.Module):
    def __init__(self, scale):
        super(ScaledNeuron, self).__init__()
        self.scale = scale
        self.t = 0
        self.neuron = neuron.IFNode(v_reset=None)
    def forward(self, x):
        # 相当于权重归一化          
        x = x / self.scale
        if self.t == 0:
            self.neuron(torch.ones_like(x)*0.5)
        x = self.neuron(x)
        self.fire_rate = torch.sum(x)/x.numel()
        self.t += 1
        return x * self.scale
    def reset(self):
        self.t = 0
        self.neuron.reset()

# SRP中的FloorLayer
class GradFloor(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

myfloor = GradFloor.apply

class ShiftNeuron(nn.Module):
    def __init__(self, scale=1., alpha=1/50000):
        super().__init__()
        self.alpha = alpha
        self.vt = 0.
        self.scale = scale
        self.neuron = neuron.IFNode(v_reset=None)
    def forward(self, x):  
        x = x / self.scale
        x = self.neuron(x)
        return x * self.scale
    def reset(self):
        if self.training:
            self.vt = self.vt + self.neuron.v.reshape(-1).mean().item()*self.alpha
        self.neuron.reset()
        if self.training == False:
            self.neuron.v = self.vt

class MyFloor(nn.Module):
    def __init__(self, up=16., t=32):
        super().__init__()
        # self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.up_val = up
        self.t = t
        self.up_init = False
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)

    def forward(self, x):
        if not self.up_init:
            self.up_init = True
            # 通道级别
            # x：[batch, channel, height, width]
            self.up = nn.Parameter((torch.ones(x.shape[1])*self.up_val).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(x.device), requires_grad=True)
        x = x / self.up
        x = myfloor(x*self.t+0.5)/self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x

class MyFloor_Layer(nn.Module):
    def __init__(self, up=8., t=32, mode='softplus'):
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t
        self.mode = mode

    def forward(self, x):
        if self.mode == 'softplus':
            x = x / F.softplus(self.up)
        else:
            x = x / self.up
        
        x = torch.clamp(x, 0, 1)
        x = myfloor(x*self.t+0.5)/self.t

        if self.mode == 'softplus':
            x = x * F.softplus(self.up)
        else:
            x = x * self.up
        return x

class MyFloor_Channel(nn.Module):
    def __init__(self, up=16., t=32, mode='softplus'):
        super().__init__()
        self.up_val = up
        self.t = t
        self.up_init = False
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.mode = mode

    def forward(self, x):
        if not self.up_init:
            self.up_init = True
            # 通道级别
            # x：[batch, channel, height, width]
            # 对x的维度做判断
            if len(x.shape) == 2:
                self.up = nn.Parameter((torch.ones(x.shape[1])*self.up_val).unsqueeze(0).to(x.device), requires_grad=True)
            elif len(x.shape) == 4:
                self.up = nn.Parameter((torch.ones(x.shape[1])*self.up_val).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(x.device), requires_grad=True)
        
        if self.mode == 'softplus':
            x = x / F.softplus(self.up)
        else:
            x = x / self.up
        x = torch.clamp(x, 0, 1)
        x = myfloor(x*self.t+0.5)/self.t
        if self.mode == 'softplus':
            x = x * F.softplus(self.up)
        else:
            x = x * self.up
        return x

class TCL(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Parameter(torch.Tensor([4.]), requires_grad=True)
    def forward(self, x):
        x = F.relu(x, inplace='True')
        x = self.up - x
        x = F.relu(x, inplace='True')
        x = self.up - x
        return x

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

# 使用SRP的MPLayer
class MPLayer(nn.Module):
    def __init__(self, v_threshold, presim_len, sim_len):
        super().__init__()
        # Spikingjelly        
        self.neuron = neuron.IFNode(v_reset=None)
        self.v_threshold = v_threshold
        self.t = 0
        self.membrane_lower = None
        self.presim_len = presim_len
        self.sim_len = sim_len
        
 
    def forward(self, x):
        with torch.no_grad():
            if self.t == 0:
                self.neuron.reset()
                self.neuron(torch.ones_like(x)*0.5)
            
            output = self.neuron(x/self.v_threshold)

            self.t += 1
            
            if self.t == self.presim_len:
                self.membrane_lower = torch.where(self.neuron.v>1e-3,torch.ones_like(output),torch.zeros_like(output))
                self.neuron.reset()
                self.neuron(torch.ones_like(x)*0.5)
            
            if self.t > self.presim_len:
                output = output * self.membrane_lower
                                                        
            if self.t == self.presim_len + self.sim_len:                     
                self.t = 0  
                    
            return output*self.v_threshold    

                           