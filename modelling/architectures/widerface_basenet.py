# ******************************************************
# Author       : liuyang
# Last modified: 2020-01-13 20:41
# Email        : gxly1314@gmail.com
# Filename     : widerface_basenet.py
# Description  : 
# ******************************************************
import torch
import torch.nn as nn
from core.workspace import register
import torch.nn.functional as F
import math

@register
class FPContext(nn.Module):
    __shared__ = ['context_range']
    def __init__(self, context_range=[5], channel_list=[256, 512, 1024, 2048], use_dilated_conv=False, use_aspp=False):
        super(FPContext, self).__init__()
        self.num_context_module = len(context_range)
        self.context_range = context_range
        self.context_modules = nn.ModuleList([])
        self.down_convs = nn.ModuleList([])
        self.use_dilated_conv = use_dilated_conv
        self.use_aspp = use_aspp
            
        #for i in range(self.num_context_module):
        for i in range(len(channel_list)):
            self.down_convs.append(nn.Conv2d(channel_list[i], 256, 1, 1, 0))

        for i in range(len(channel_list)):
            self.context_modules.append(nn.ModuleList([]))
            for j in range(len(self.context_range)):
                self.context_modules[i].append(nn.Conv2d(256, 256, 3, 1, 1))

    def forward(self, feature_list):
        fp_context_fts = []
        for i in range(len(feature_list)):
            down_conv_ft = self.down_convs[i](feature_list[i])
            fp_context_list = []
            for layer in self.context_modules[i]:
                fp_context_list.append(layer(down_conv_ft))
            fp_context_fts.append(fp_context_list)

        return fp_context_fts


@register
class WiderFaceBaseNetFPContext(nn.Module):
    __shared__ = ['phase']
    __inject__ = ['backbone', 'fpn', 'pred_net', 'fp_context', 'pred_net_1']
    def __init__(self, backbone, fpn, pred_net, pred_net_1, fp_context=None, phase='training', out_bb_ft=False):
        super(WiderFaceBaseNetFPContext, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.pred_net = pred_net
        self.pred_net_1 = pred_net_1
        self.phase = phase
        self.out_bb_ft = out_bb_ft

        self.fp_context = fp_context        
        
        if self.phase == 'training':
            self.fpn.weight_init()
            self.pred_net.weight_init()
            self.pred_net_1.weight_init()

    def forward(self, x):
        feature_list = self.backbone(x)
        if self.fp_context is not None:
            fp_context_fts = self.fp_context(feature_list)

        pyramid_feature_list = self.fpn(feature_list)

        if self.phase == 'training':
            conf, loc, mask_fp_context_fts = self.pred_net(pyramid_feature_list)
            conf_1 = self.pred_net_1(pyramid_feature_list, fp_context_fts, mask_fp_context_fts) 
            if self.fp_context is not None:
                return conf, loc, conf_1
            else:
                return conf, loc
        else:
            conf, loc, mask_fp_context_fts = self.pred_net(pyramid_feature_list)
            conf_1 = self.pred_net_1(pyramid_feature_list, fp_context_fts, mask_fp_context_fts) 
            if self.fp_context is not None:
                return conf, loc, conf_1
            else:
                return conf, loc


@register
class WiderFaceBaseNet(nn.Module):
    __shared__ = ['phase']
    __inject__ = ['backbone', 'fpn', 'pred_net']
    def __init__(self, backbone, fpn, pred_net, phase='training', out_bb_ft=False):
        super(WiderFaceBaseNet, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.pred_net = pred_net
        self.phase = phase
        self.out_bb_ft = out_bb_ft
        if self.phase == 'training':
            self.fpn.weight_init()
            self.pred_net.weight_init()

    def forward(self, x):
        feature_list = self.backbone(x)
        fpn_list= self.fpn(feature_list)
        if len(fpn_list) == 2:
            pyramid_feature_list = fpn_list[0]
            dsfd_ft_list = fpn_list[1]
        else:
            pyramid_feature_list = fpn_list
        if self.phase == 'training':
            if len(fpn_list) == 2:
                conf, loc, dsfd_conf, dsfd_loc = self.pred_net(pyramid_feature_list, dsfd_ft_list)
                return conf, loc, dsfd_conf, dsfd_loc
            conf, loc = self.pred_net(pyramid_feature_list)
            if self.out_bb_ft:
                return conf, loc, feature_list
            else:
                return conf, loc
        else:
            if len(fpn_list) == 2:
                conf, loc = self.pred_net(pyramid_feature_list, dsfd_ft_list)
                return conf, loc 
            conf, loc = self.pred_net(pyramid_feature_list)
            if self.out_bb_ft:
                return conf, loc, feature_list
            else:
                return conf, loc
