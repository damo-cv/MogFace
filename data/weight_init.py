# ******************************************************
# Author        : liuyang
# Last modified : 2020-01-13 20:55
# Email         : gxly1314@gmail.com
# Filename      : weight_init.py
# Description   : 
# ******************************************************
import torch.nn as nn
from core.workspace import register
import numpy as np
import torch

@register
class NormalWeightInit(object):
    def __call__(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()

        if isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.01)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data[...] = 1
            m.bias.data.zero_()

@register
class RetinaClsWeightInit(object):
    def __call__(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)
            # debug finished!
            if 'bias' in m.state_dict().keys():
                pi = 0.01
                #bv = -np.ones(num_scales * num_ratios * num_class) * np.log((1 - pi) / pi)
                # TODO check if need num_anchors
                bv = -np.ones(1) * np.log((1 - pi) / pi)
                m.bias.data = (torch.from_numpy(bv)).expand_as(m.bias.data).float().cuda()


