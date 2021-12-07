# ******************************************************
# Author        : liuyang
# Last modified :	2020-01-13 20:44
# Email         : gxly1314@gmail.com
# Filename      :	pred_net.py
# Description   : 
# ******************************************************
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from core.workspace import register

class conv_bn(nn.Module):
    """docstring for conv"""

    def __init__(self,
                 in_plane,
                 out_plane,
                 kernel_size,
                 stride,
                 padding):
        super(conv_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_plane, out_plane,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_plane)

    def forward(self, x):
        x = self.conv1(x)
        return self.bn1(x)

class CPM(nn.Module):
    """docstring for CPM"""

    def __init__(self, in_plane):
        super(CPM, self).__init__()
        self.branch1 = conv_bn(in_plane, 512, 1, 1, 0)
        self.branch2a = conv_bn(in_plane, 128, 1, 1, 0)
        self.branch2b = conv_bn(128, 128, 3, 1, 1)
        self.branch2c = conv_bn(128, 512, 1, 1, 0)

        self.ssh_1 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.ssh_dimred = nn.Conv2d(
            512, 64, kernel_size=3, stride=1, padding=1)
        self.ssh_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.ssh_3a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.ssh_3b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out_residual = self.branch1(x)
        x = F.relu(self.branch2a(x), inplace=True)
        x = F.relu(self.branch2b(x), inplace=True)
        x = self.branch2c(x)

        rescomb = F.relu(x + out_residual, inplace=True)
        ssh1 = self.ssh_1(rescomb)
        ssh_dimred = F.relu(self.ssh_dimred(rescomb), inplace=True)
        ssh_2 = self.ssh_2(ssh_dimred)
        ssh_3a = F.relu(self.ssh_3a(ssh_dimred), inplace=True)
        ssh_3b = self.ssh_3b(ssh_3a)

        ssh_out = torch.cat([ssh1, ssh_2, ssh_3b], dim=1)
        ssh_out = F.relu(ssh_out, inplace=True)
        return ssh_out

class SSHContext(nn.Module):
    def __init__(self, channels, Xchannels=256):
        super(SSHContext, self).__init__()

        self.conv1 = nn.Conv2d(channels,Xchannels,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(channels,Xchannels//2,kernel_size=3,dilation=2,stride=1,padding=2)
        self.conv2_1 = nn.Conv2d(Xchannels//2,Xchannels//2,kernel_size=3,stride=1,padding=1)
        self.conv2_2 = nn.Conv2d(Xchannels//2,Xchannels//2,kernel_size=3,dilation=2,stride=1,padding=2)
        self.conv2_2_1 = nn.Conv2d(Xchannels//2,Xchannels//2,kernel_size=3,stride=1,padding=1)
        

    def forward(self, x):
        x1 = F.relu(self.conv1(x),inplace=True)
        x2 = F.relu(self.conv2(x),inplace=True)
        x2_1 = F.relu(self.conv2_1(x2),inplace=True)
        x2_2 = F.relu(self.conv2_2(x2),inplace=True)
        x2_2 = F.relu(self.conv2_2_1(x2_2),inplace=True)

        return torch.cat([x1,x2_1,x2_2],1)

class DeepHead(nn.Module):
    def __init__(self, in_channel=256, out_channel=256, use_gn=False, num_conv=4):
        super(DeepHead, self).__init__()
        self.use_gn = use_gn
        self.num_conv = num_conv
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        if self.use_gn:
            self.gn1 = nn.GroupNorm(16, out_channel)
            self.gn2 = nn.GroupNorm(16, out_channel)
            self.gn3 = nn.GroupNorm(16, out_channel)
            self.gn4 = nn.GroupNorm(16, out_channel)

    def forward(self, x):
        if self.use_gn:
            x1 = F.relu(self.gn1(self.conv1(x)),inplace=True)
            x2 = F.relu(self.gn2(self.conv1(x1)),inplace=True)
            x3 = F.relu(self.gn3(self.conv1(x2)),inplace=True)
            x4 = F.relu(self.gn4(self.conv1(x3)),inplace=True)
        else:
            x1 = F.relu(self.conv1(x),inplace=True)
            x2 = F.relu(self.conv1(x1),inplace=True)
            if self.num_conv == 2:
                return x2
            x3 = F.relu(self.conv1(x2),inplace=True)
            x4 = F.relu(self.conv1(x3),inplace=True)

        return x4

@register
class RetinaPredNetFPContext_1(nn.Module):
    __shared__ = ['num_classes', 'weight_init_fn', 'phase']
    __inject__ = ['weight_init_fn', 'retina_cls_weight_init_fn']
    def __init__(self, num_anchor_per_pixel=1, num_classes=1, \
                 input_ch_list=[256, 256, 256, 256, 256, 256], \
                 weight_init_fn=None, retina_cls_weight_init_fn=None, \
                 phase='training', use_pyramid_ft=True):
        super(RetinaPredNetFPContext_1, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.weight_init_fn = weight_init_fn
        self.retina_cls_weight_init_fn = retina_cls_weight_init_fn
        self.use_pyramid_ft = use_pyramid_ft

        cls_br_list = []
        loc_br_list = []
        for i in range(6):
            cls_br_list.append(
               nn.Conv2d(input_ch_list[i], 1 * num_anchor_per_pixel, kernel_size=3, \
                         stride=1, padding=1)
            )

        self.pred_cls = nn.ModuleList(cls_br_list)

        self.sigmoid = nn.Sigmoid()

    def forward(self, pyramid_feature_list, fp_context_fts, mask_fp_context_fts):
        feature_context_list = []
        for i in range(len(fp_context_fts)):
            mask_fp_context_feature = mask_fp_context_fts[i]
            fp_context = fp_context_fts[i]
            fp_context_feature = None
            for j in range(len(fp_context)):
                if j == 0:
                    fp_context_feature = fp_context[j] * mask_fp_context_feature[j]
                else:
                    fp_context_feature += fp_context[j] * mask_fp_context_feature[j]
            if self.use_pyramid_ft:
                feature_context_list.append(fp_context_feature + pyramid_feature_list[i])
            else:
                feature_context_list.append(fp_context_feature)

        feature_context_list.append(pyramid_feature_list[-2])
        feature_context_list.append(pyramid_feature_list[-1])

        conf = []
        for (x,c) in zip(feature_context_list, self.pred_cls):
            conf.append(c(x).permute(0,2,3,1).contiguous())

        conf = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in conf], 1)

        if self.phase == 'training':
            output = conf.view(conf.size(0), -1, self.num_classes)

            return output
        else:
            output = self.sigmoid(conf.view(conf.size(0), -1, self.num_classes))
            
            return output

    def weight_init(self):
        #pass
        for layer in self.pred_cls.modules():
            layer.apply(self.retina_cls_weight_init_fn)

@register
class RetinaPredNetFPContext(nn.Module):
    __shared__ = ['num_classes', 'weight_init_fn', 'phase', 'context_range', 'fp_th']
    __inject__ = ['weight_init_fn', 'retina_cls_weight_init_fn']
    def __init__(self, num_anchor_per_pixel=1, num_classes=1, \
                 input_ch_list=[256, 256, 256, 256, 256, 256], \
                 weight_init_fn=None, retina_cls_weight_init_fn=None, \
                 phase='training', use_mask_fp_context=False, fp_th=0.5,
                 context_range=[5], mask_fp_version='hard_attention',
                 ):
        super(RetinaPredNetFPContext, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.weight_init_fn = weight_init_fn
        self.retina_cls_weight_init_fn = retina_cls_weight_init_fn
        self.use_mask_fp_context = use_mask_fp_context
        self.mask_fp_version = mask_fp_version

        if self.use_mask_fp_context:
            self.fp_th = fp_th
            self.context_range = context_range

        cls_br_list = []
        loc_br_list = []
        for i in range(6):
            cls_br_list.append(
               nn.Conv2d(input_ch_list[i], 1 * num_anchor_per_pixel, kernel_size=3, \
                         stride=1, padding=1)
            )

            loc_br_list.append(
               nn.Conv2d(input_ch_list[i], 4 * num_anchor_per_pixel, kernel_size=3, \
                         stride=1, padding=1)
            )

        self.pred_cls = nn.ModuleList(cls_br_list)
        self.pred_loc = nn.ModuleList(loc_br_list)

        self.sigmoid = nn.Sigmoid()

    def forward(self, pyramid_feature_list, targets=None):
        loc = []
        conf = []
        if self.use_mask_fp_context:
            ret_mask_fp_context_fts = []

        for (x,l,c) in zip(pyramid_feature_list, self.pred_loc, self.pred_cls):
            loc_x = l(x)
            conf_x = c(x)
            if self.use_mask_fp_context:
                if self.mask_fp_version == 'hard_attention':
                    with torch.no_grad():
                        tmp_h = conf_x.shape[2]
                        tmp_w = conf_x.shape[3]

                        conf_sigmoid = F.sigmoid(conf_x)
                        #fp_th = self.fp_th

                        mask_pos_predict_anchor = conf_sigmoid > self.fp_th
                        mask_pos = torch.where(mask_pos_predict_anchor == 1)
                        mask_pos = list(mask_pos)

                        mask_fp_context_fts = []

                        shift_lists = []
                        # generate two context idx around mask_pos
                        for i in range(len(self.context_range)):
                            context_size = self.context_range[i]
                            max_shift = int((context_size - 1) / 2)
                            shift_list = []
                            for tmp_i in range(-1 * max_shift, max_shift + 1):
                                for tmp_j in range(-1 * max_shift, max_shift + 1):
                                    if tmp_i == 0 and tmp_j == 0:
                                        continue
                                    shift_list.append([tmp_i, tmp_j])
                            shift_lists.append(shift_list)

                        for shift_list in shift_lists:
                            mask_fp_context_ft = mask_pos_predict_anchor.clone()
                            for shift in shift_list:
                                tmp_shift = (mask_pos[0], mask_pos[1], \
                                        torch.clamp(mask_pos[2] + shift[0], 0, tmp_h), 
                                        torch.clamp(mask_pos[3] + shift[1], 0, tmp_w))
                                mask_fp_context_ft[tmp_shift] = 1
                                
                            mask_fp_context_fts.append(mask_fp_context_ft)

                    ret_mask_fp_context_fts.append(mask_fp_context_fts)

            loc.append(loc_x.permute(0,2,3,1).contiguous())
            conf.append(conf_x.permute(0,2,3,1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1, 4) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in conf], 1)

        if self.phase == 'training':
            if self.use_mask_fp_context:
                output = (
                         conf.view(conf.size(0), -1, self.num_classes), \
                         loc.view(loc.size(0), -1, 4), \
                         ret_mask_fp_context_fts,
                         )
            else:
                output = (
                         conf.view(conf.size(0), -1, self.num_classes), \
                         loc.view(loc.size(0), -1, 4)
                         )
            return output
        else:
            if self.use_mask_fp_context:
                output = (
                        self.sigmoid(conf.view(conf.size(0), -1, self.num_classes)),
                        loc.view(loc.size(0), -1, 4),
                        ret_mask_fp_context_fts
                        )
            else:
                output = (
                        self.sigmoid(conf.view(conf.size(0), -1, self.num_classes)),
                        loc.view(loc.size(0), -1, 4),
                        )
            
        return output

    def weight_init(self):
        for layer in self.pred_cls.modules():
            layer.apply(self.retina_cls_weight_init_fn)
        for layer in self.pred_loc.modules():
            layer.apply(self.weight_init_fn)

       

@register
class RetinaPredNet(nn.Module):
    __shared__ = ['num_classes', 'weight_init_fn', 'phase']
    __inject__ = ['weight_init_fn', 'retina_cls_weight_init_fn']
    def __init__(self, num_anchor_per_pixel=1, num_classes=1, \
                 input_ch_list=[256, 256, 256, 256, 256, 256], \
                 weight_init_fn=None, retina_cls_weight_init_fn=None, \
                 use_deep_head=False, deep_head_with_gn=False, use_ssh=False, \
                 use_cpm=False,  phase='training'):
        super(RetinaPredNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.weight_init_fn = weight_init_fn
        self.retina_cls_weight_init_fn = retina_cls_weight_init_fn

        self.use_ssh = use_ssh
        self.use_cpm = use_cpm

        self.use_deep_head  = use_deep_head
        self.deep_head_with_gn = deep_head_with_gn

        if self.use_ssh:
            self.conv_SSH = SSHContext(input_ch_list[0], self.deep_head_ch // 2)
        if self.use_cpm:
            self.conv_CPM = CPM(input_ch_list[0])

        if self.use_deep_head:
            if self.deep_head_with_gn:
                self.deep_loc_head = DeepHead(256, 256, use_gn=True)
                self.deep_cls_head = DeepHead(256, 256, use_gn=True)
            else:
                self.deep_loc_head = DeepHead(256, 256)
                self.deep_cls_head = DeepHead(256, 256)
            # share pred net
            self.pred_cls = nn.Conv2d(input_ch_list[0], 1 * num_anchor_per_pixel, 3, 1, 1)
            self.pred_loc = nn.Conv2d(input_ch_list[0], 4 * num_anchor_per_pixel, 3, 1, 1)
        else:
            cls_br_list = []
            loc_br_list = []
            for i in range(6):
                cls_br_list.append(
                   nn.Conv2d(input_ch_list[i], 1 * num_anchor_per_pixel, kernel_size=3, \
                             stride=1, padding=1)
                )

                loc_br_list.append(
                   nn.Conv2d(input_ch_list[i], 4 * num_anchor_per_pixel, kernel_size=3, \
                             stride=1, padding=1)
                )

            self.pred_cls = nn.ModuleList(cls_br_list)
            self.pred_loc = nn.ModuleList(loc_br_list)

        self.sigmoid = nn.Sigmoid()

    def forward(self, pyramid_feature_list):
        loc = []
        conf = []
        if self.use_deep_head:
            for x in pyramid_feature_list:
                if self.use_ssh:
                    x = self.conv_SSH(x)
                if self.use_cpm:
                    x = self.conv_CPM(x)
                x_cls = self.deep_cls_head(x)
                x_loc = self.deep_loc_head(x)
                conf.append(self.pred_cls(x_cls).permute(0,2,3,1).contiguous())
                loc.append(self.pred_loc(x_loc).permute(0,2,3,1).contiguous())
        else:
            for (x,l,c) in zip(pyramid_feature_list, self.pred_loc, self.pred_cls):
                loc.append(l(x).permute(0,2,3,1).contiguous())
                conf.append(c(x).permute(0,2,3,1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1, 4) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in conf], 1)

        if self.phase == 'training':
            output = (
                     conf.view(conf.size(0), -1, self.num_classes), \
                     loc.view(loc.size(0), -1, 4)
                     )
            return output
        else:
            output = (
                    self.sigmoid(conf.view(conf.size(0), -1, self.num_classes)),
                    loc.view(loc.size(0), -1, 4),
                    )
            
        return output

    def weight_init(self):
        for layer in self.pred_cls.modules():
            layer.apply(self.retina_cls_weight_init_fn)
        for layer in self.pred_loc.modules():
            layer.apply(self.weight_init_fn)

@register
class MogPredNet(nn.Module):
    __shared__ = ['num_classes', 'weight_init_fn', 'phase']
    __inject__ = ['weight_init_fn', 'retina_cls_weight_init_fn']
    def __init__(self, num_anchor_per_pixel=1, num_classes=1, \
                 input_ch_list=[256, 256, 256, 256, 256, 256], \
                 weight_init_fn=None, retina_cls_weight_init_fn=None, \
                 use_deep_head=False, phase='training',  deep_head_with_gn=False, use_ssh = False, use_cpm=False, use_dsfd=False, deep_head_ch=256):
        super(MogPredNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.weight_init_fn = weight_init_fn
        self.retina_cls_weight_init_fn = retina_cls_weight_init_fn
        self.use_deep_head  = use_deep_head
        self.deep_head_with_gn = deep_head_with_gn

        self.use_ssh = use_ssh
        self.use_cpm = use_cpm
        self.use_dsfd = use_dsfd

        self.deep_head_ch = deep_head_ch

        if self.use_dsfd:
            self.dsfd_loc = nn.Conv2d(input_ch_list[0],4,kernel_size=3,stride=1,padding=1)
            self.dsfd_conf = nn.Conv2d(input_ch_list[0],1,kernel_size=3,stride=1,padding=1)

        if self.use_ssh:
            self.conv_SSH = SSHContext(input_ch_list[0], self.deep_head_ch // 2)

        if self.use_cpm:
            self.conv_CPM = CPM(input_ch_list[0])

        if self.use_deep_head:
            if self.deep_head_with_gn:
                self.deep_loc_head = DeepHead(self.deep_head_ch, self.deep_head_ch, use_gn=True)
                self.deep_cls_head = DeepHead(self.deep_head_ch, self.deep_head_ch, use_gn=True)
            else:
                self.deep_loc_head = DeepHead(self.deep_head_ch, self.deep_head_ch)
                self.deep_cls_head = DeepHead(self.deep_head_ch, self.deep_head_ch)
            # share pred net
            self.pred_cls = nn.Conv2d(self.deep_head_ch, 1 * num_anchor_per_pixel, 3, 1, 1)
            self.pred_loc = nn.Conv2d(self.deep_head_ch, 4 * num_anchor_per_pixel, 3, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, pyramid_feature_list, dsfd_ft_list=None):
        loc = []
        conf = []

        if dsfd_ft_list is not None:
            assert self.use_dsfd == True, "should set 'use_dsfd = True'."
            dsfd_conf = []
            dsfd_loc = []
            for x in dsfd_ft_list:
                dsfd_conf.append(self.dsfd_conf(x).permute(0,2,3,1).contiguous())
                dsfd_loc.append(self.dsfd_loc(x).permute(0,2,3,1).contiguous())

        if self.use_deep_head:
            for x in pyramid_feature_list:
                if self.use_ssh:
                    x = self.conv_SSH(x)
                if self.use_cpm:
                    x = self.conv_CPM(x)
                x_cls = self.deep_cls_head(x)
                x_loc = self.deep_loc_head(x)

                conf.append(self.pred_cls(x_cls).permute(0,2,3,1).contiguous())
                loc.append(self.pred_loc(x_loc).permute(0,2,3,1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1, 4) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in conf], 1)
        if self.use_dsfd:
            dsfd_loc = torch.cat([o.view(o.size(0), -1, 4) for o in dsfd_loc], 1)
            dsfd_conf = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in dsfd_conf], 1)

        if self.phase == 'training':
            if self.use_dsfd:
                output = (
                         conf.view(conf.size(0), -1, self.num_classes), \
                         loc.view(loc.size(0), -1, 4),
                         dsfd_conf.view(dsfd_conf.size(0), -1, self.num_classes),
                         dsfd_loc.view(dsfd_loc.size(0), -1, 4),
                         )
            else:
                output = (
                         conf.view(conf.size(0), -1, self.num_classes), \
                         loc.view(loc.size(0), -1, 4)
                         )
            return output
        else:
            output = (
                    self.sigmoid(conf.view(conf.size(0), -1, self.num_classes)),
                    loc.view(loc.size(0), -1, 4),
                    )
            
        return output

    def weight_init(self):
        for layer in self.modules():
            layer.apply(self.weight_init_fn)
        for layer in self.pred_cls.modules():
            layer.apply(self.retina_cls_weight_init_fn)
        if self.use_dsfd:
            for layer in self.dsfd_conf.modules():
                layer.apply(self.retina_cls_weight_init_fn)
