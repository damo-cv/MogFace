# ******************************************************
# Author        : liuyang
# Last modified :	2020-01-13 20:44
# Email         : gxly1314@gmail.com
# Filename      :	loss.py
# Description   : 
# ******************************************************
import torch.nn as nn
import torch
from core.workspace import register

@register
class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_label=-1, eps=1e-7):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_label = ignore_label
        self.eps = eps
            
    def __call__(self, pred, target):
        pred = pred.sigmoid()
        mask = 1 - target.eq(self.ignore_label).float()
        pos_part = target.float() * ((pred + self.eps).log())
        neg_part = (1 - target).float() * ((1 - pred + self.eps).log())
        loss = -(pos_part + neg_part) * mask.float()
        return loss.sum() / max(target.shape[0], (target >= 0).sum())


@register
class FocalLossFewSample(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_label=-1, eps=1e-7, few_sample=False):
        super(FocalLossFewSample, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_label = ignore_label
        self.eps = eps
        self.few_sample = few_sample
            
    def __call__(self, pred, target):
        pred = pred.sigmoid()
        mask = 1 - target.eq(self.ignore_label).float()
        pos_part = (1 - pred).pow(self.gamma) * target.float() * ((pred + self.eps).log())
        neg_part = pred.pow(self.gamma) * (1 - target).float() * ((1 - pred + self.eps).log())
        loss = -(self.alpha * pos_part + (1 - self.alpha) * neg_part) * mask.float()
        if self.few_sample:
            return loss.sum() / max(target.shape[0], (target >= 0).sum())
        else:
            return loss.sum() / max(target.shape[0], (target > 0).sum())
@register
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_label=-1, eps=1e-7, few_sample=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_label = ignore_label
        self.eps = eps
        self.few_sample = few_sample
            
    def __call__(self, pred, target):
        pred = pred.sigmoid()
        mask = 1 - target.eq(self.ignore_label).float()
        pos_part = (1 - pred).pow(self.gamma) * target.float() * ((pred + self.eps).log())
        neg_part = pred.pow(self.gamma) * (1 - target).float() * ((1 - pred + self.eps).log())
        loss = -(self.alpha * pos_part + (1 - self.alpha) * neg_part) * mask.float()
        if self.few_sample:
            return loss.sum() / max(target.shape[0], (target >= 0).sum())
        else:
            return loss.sum() / max(target.shape[0], (target > 0).sum())

@register
class SmoothL1Loss(nn.Module):
    def __init__(self, sigma=3):
        super(SmoothL1Loss, self).__init__()
        self.sigma = sigma

    def __call__(self, loc, bbox_targets):
        '''
        loc [batch_size, num_anchors, 4]
        bbox_targets [batch_size, num_anchors, 5]
        '''
        pos = (bbox_targets[:, : ,-1] > 0)
        pos_idx = pos.unsqueeze(2).expand_as(loc)

        pred = loc[pos_idx].view(-1,4)
        gt = bbox_targets[:, : , :4][pos_idx].view(-1,4)

        sigma2 = self.sigma ** 2
        cond_point = 1 / sigma2
        x = pred - gt
        abs_x = torch.abs(x)
        in_mask = abs_x < cond_point
        out_mask = 1 - in_mask.float()
        in_value = 0.5 * (self.sigma * x) ** 2
        out_value = abs_x - 0.5 /  sigma2 
        value = in_value * in_mask.type_as(in_value) + out_value * out_mask.type_as(out_value)

        return value.sum() / max(bbox_targets.shape[0], pos.sum())
