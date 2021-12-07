# ******************************************************
# Author        : liuyang
# Last modified : 2020-01-13 20:55
# Email         : gxly1314@gmail.com
# Filename      : optimizer.py
# Description   : 
# ******************************************************
import torch.optim as optim
from core.workspace import register
import torch

@register
class ClipGradients(object):

    def __init__(self, max_norm=5, norm_type=2):
        self.max_norm = max_norm
        self.norm_type = norm_type
        
    def __call__(self, parameters):
        torch.nn.utils.clip_grad_norm_(parameters, self.max_norm, self.norm_type)
@register
class SGDOptimizer(object):
    __shared__ = ['lr']
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=1e-4):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = float(weight_decay)
    def __call__(self, net):
        optimizer = optim.SGD(net.parameters(), \
                              lr=self.lr, \
                              momentum=self.momentum, \
                              weight_decay=self.weight_decay)
        return optimizer

@register
class LinearWarmup(object):
    __shared__ = ['lr']
    def __init__(self, warm_up_iter=3000, ini_lr=1e-6, lr=0.01, \
                 step_list=[110000, 140000, 160000], gamma=0.1):
        self.warm_up_iter = warm_up_iter
        self.step_list = step_list
        self.lr = lr
        self.ini_lr = float(ini_lr)
        self.gamma = gamma

    def __call__(self, optimizer, cur_iter_num):
        step_idx = 0
        for idx in range(len(self.step_list)):
            if cur_iter_num < self.step_list[idx]:
                step_idx = idx
                break
        if cur_iter_num < self.warm_up_iter:
            lr = self.ini_lr + (self.lr - self.ini_lr) * cur_iter_num / self.warm_up_iter
        else:
            lr = self.lr * (self.gamma ** (step_idx))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
