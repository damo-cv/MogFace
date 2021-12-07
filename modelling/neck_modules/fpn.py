# ******************************************************
# Author        : liuyang
# Last modified : 2020-01-13 20:43
# Email         : gxly1314@gmail.com
# Filename      : fpn.py
# Description   : 
# ******************************************************
import torch.nn as nn
import torch.nn.functional as F
import torch
from core.workspace import register

class FeatureFusion(nn.Module):
    def __init__(self, lat_ch=256, **channels):
        super(FeatureFusion, self).__init__()
        self.main_conv = nn.Conv2d(channels['main'], lat_ch, kernel_size=1)
        

    def forward(self,up,main):
        main = self.main_conv(main)
        _,_,H,W = main.size()
        res = F.upsample(up,scale_factor=2,mode='bilinear') 
        if res.size(2) != main.size(2) or res.size(3) != main.size(3):
            res = res[:,:,0:H,0:W]
        res = res + main
        return res

@register
class LFPN(nn.Module):
    __inject__ = ['weight_init_fn']
    __shared__ = ['weight_init_fn']

    def __init__(self, weight_init_fn=None, c2_out_ch=256, c3_out_ch=512, c4_out_ch=1024, c5_out_ch=2048, \
                 c6_mid_ch=512, c6_out_ch=512, c7_mid_ch=128, c7_out_ch=256, out_dsfd_ft=False):
        super(LFPN, self).__init__()
        self.weight_init_fn = weight_init_fn
        self.out_dsfd_ft = out_dsfd_ft
        if self.out_dsfd_ft:
            dsfd_module = []
            dsfd_module.append(nn.Conv2d(256,256,kernel_size=3,padding=1))
            dsfd_module.append(nn.Conv2d(512,256,kernel_size=3,padding=1))
            dsfd_module.append(nn.Conv2d(1024,256,kernel_size=3,padding=1))
            dsfd_module.append(nn.Conv2d(2048,256,kernel_size=3,padding=1))
            dsfd_module.append(nn.Conv2d(256,256,kernel_size=3,padding=1))
            dsfd_module.append(nn.Conv2d(256,256,kernel_size=3,padding=1))
            self.dsfd_modules = nn.ModuleList(dsfd_module)

        c6_input_ch = c5_out_ch
        self.c6= nn.Sequential(                                        
            *[nn.Conv2d(c6_input_ch, c6_mid_ch, kernel_size=1,),      
                nn.BatchNorm2d(c6_mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(c6_mid_ch, c6_out_ch, kernel_size=3,padding=1,stride=2),
                nn.BatchNorm2d(c6_out_ch),
                nn.ReLU(inplace=True)]
            )
        self.c7 = nn.Sequential(
            *[nn.Conv2d(c6_out_ch, c7_mid_ch, kernel_size=1,),
                nn.BatchNorm2d(c7_mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(c7_mid_ch, c7_out_ch, kernel_size=3,padding=1,stride=2),
                nn.BatchNorm2d(c7_out_ch),
                nn.ReLU(inplace=True)]
            )

        self.p2_lat = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.p3_lat = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.p4_lat = nn.Conv2d(256,256,kernel_size=3,padding=1)

        self.c5_lat = nn.Conv2d(c6_input_ch,256,kernel_size=3,padding=1)
        self.c6_lat = nn.Conv2d(c6_out_ch,256,kernel_size=3,padding=1)
        self.c7_lat = nn.Conv2d(c7_out_ch,256,kernel_size=3,padding=1)

        self.ff_c5_c4 = FeatureFusion(main=c4_out_ch)
        self.ff_c4_c3 = FeatureFusion(main=c3_out_ch)
        self.ff_c3_c2 = FeatureFusion(main=c2_out_ch)

    def forward(self, feature_list):
        c2, c3, c4, c5 = feature_list
        c6 = self.c6(c5)
        c7 = self.c7(c6)

        c5 = self.c5_lat(c5)
        c6 = self.c6_lat(c6)
        c7 = self.c7_lat(c7)
    
        if self.out_dsfd_ft:
            dsfd_fts = []
            dsfd_fts.append(self.dsfd_modules[0](c2))
            dsfd_fts.append(self.dsfd_modules[1](c3))
            dsfd_fts.append(self.dsfd_modules[2](c4))
            dsfd_fts.append(self.dsfd_modules[3](feature_list[-1]))
            dsfd_fts.append(self.dsfd_modules[4](c6))
            dsfd_fts.append(self.dsfd_modules[5](c7))

        p4 = self.ff_c5_c4(c5, c4)
        p3 = self.ff_c4_c3(p4, c3)
        p2 = self.ff_c3_c2(p3, c2)

        p2 = self.p2_lat(p2)
        p3 = self.p3_lat(p3)
        p4 = self.p4_lat(p4)

        if self.out_dsfd_ft:
            return ([p2, p3, p4, c5, c6, c7], dsfd_fts)
        else:
            return ([p2, p3, p4, c5, c6, c7])

    def weight_init(self):
        for layer in self.modules():
            layer.apply(self.weight_init_fn)

@register
class LightFPN(nn.Module):
    __inject__ = ['weight_init_fn']
    __shared__ = ['weight_init_fn']

    def __init__(self, weight_init_fn=None, c2_out_ch=256, c3_out_ch=512, c4_out_ch=1024, c5_out_ch=2048, \
                 c6_mid_ch=512, c6_out_ch=512, c7_mid_ch=128, c7_out_ch=256, lat_ch=16, out_dsfd_ft=False, use_p2=True, ret_4_layer=False, ret_6_layer=False):
        super(LightFPN, self).__init__()
        self.weight_init_fn = weight_init_fn
        self.use_p2 = use_p2
        self.ret_4_layer = ret_4_layer
        self.ret_6_layer = ret_6_layer

        self.c6= nn.Sequential(                                        
            *[nn.Conv2d(c5_out_ch, lat_ch, kernel_size=1,),      
                nn.BatchNorm2d(lat_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(lat_ch, lat_ch, kernel_size=3,padding=1,stride=2),
                nn.BatchNorm2d(lat_ch),
                nn.ReLU(inplace=True)]
            )
        self.c7 = nn.Sequential(
            *[nn.Conv2d(lat_ch, lat_ch, kernel_size=1,),
                nn.BatchNorm2d(lat_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(lat_ch, lat_ch, kernel_size=3,padding=1,stride=2),
                nn.BatchNorm2d(lat_ch),
                nn.ReLU(inplace=True)]
            )

        self.c4_lat = nn.Conv2d(c4_out_ch, lat_ch,kernel_size=3,padding=1)
        self.c5_lat = nn.Conv2d(c5_out_ch, lat_ch,kernel_size=3,padding=1)

        self.ff_c5_c4 = FeatureFusion(main=c4_out_ch, lat_ch=lat_ch)
        self.ff_c4_c3 = FeatureFusion(main=c3_out_ch, lat_ch=lat_ch)
        self.ff_c3_c2 = FeatureFusion(main=c2_out_ch, lat_ch=lat_ch)

    def forward(self, feature_list):
        c2, c3, c4, or_c5 = feature_list
        c5 = self.c5_lat(or_c5)

        p4 = self.ff_c5_c4(c5, c4)
        p3 = self.ff_c4_c3(p4, c3)
        if not self.use_p2:
            return ([p3, p4, c5])
        p2 = self.ff_c3_c2(p3, c2)
        if self.ret_4_layer:
            return ([p2, p3, p4, c5])
        if self.ret_6_layer:
            c6 = self.c6(or_c5)
            c7 = self.c7(c6)
            return ([p2, p3, p4, c5, c6, c7])

        return ([p2, p3, p4])

    def weight_init(self):
        for layer in self.modules():
            layer.apply(self.weight_init_fn)
