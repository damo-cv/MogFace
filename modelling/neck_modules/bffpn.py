# ******************************************************
# Author        : liuyang
# Last modified : 2020-01-13 20:43
# Email         : gxly1314@gmail.com
# Filename      : fpn.py
# Description   : 
# ******************************************************
import torch.nn as nn
import torch.nn.functional as F
from core.workspace import register
import torch

class ContextTexture (nn.Module):
    """docstring for ContextTexture """
    def __init__(self, use_attention=True):
        super(ContextTexture , self).__init__()
        self.use_attention = use_attention

    def forward(self,up,main,attention):
        # up = self.up_conv(up)
        _,_,H_up,_ = up.size()
        _,C,H,W = main.size()
        ratio = max(round(H / H_up), round(H_up / H))
        high_resolution = max(H, H_up)
        low_resolution = min(H, H_up)
        ratio = 1
        while high_resolution != low_resolution:
            high_resolution = (high_resolution + 1) // 2
            ratio = ratio * 2
            if abs(high_resolution - low_resolution) <= 1:
                break
            #print ('resolution, ',  high_resolution, low_resolution)
            if high_resolution == 1:
                import pdb;pdb.set_trace()
        ratio = 1 / ratio if H / H_up <= 1 else ratio

        if ratio > 1:
            res = F.upsample(up,scale_factor=ratio,mode='bilinear') 
        elif ratio < 1:
            res = F.max_pool2d(up, kernel_size=int(1/ratio))
        if res.size(2) <  main.size(2):
            pad_res = torch.zeros(main.shape).cuda(res.device)
            pad_res[:, :, : res.size(2), :] = res
            res = pad_res
        if res.size(2) >  main.size(2):
            res = res[:,:,0:main.size(2),:]
        if res.size(3) <  main.size(3):
            pad_res = torch.zeros(main.shape).cuda(res.device)
            pad_res[:, :, :, :res.size(3)] = res
            attention = attention[:,:,:,:res.size(3)]
            res = pad_res
        if res.size(3) > main.size(3):
            res = res[:,:,:,:main.size(3)]
        if self.use_attention:
            res = attention * res + main
        else:
            res = res + main
        return res

@register
class BFFPN(nn.Module):
    __inject__ = ['weight_init_fn']
    __shared__ = ['weight_init_fn']

    def __init__(self, weight_init_fn=None, c2_out_ch=256, c3_out_ch=512, c4_out_ch=1024, c5_out_ch=2048, \
                 c6_mid_ch=512, c6_out_ch=512, c7_mid_ch=128, c7_out_ch=256, fpn_architecture=None, use_attention=True):
        super(BFFPN, self).__init__()

        self.weight_init_fn = weight_init_fn
        self.use_attention = use_attention
        c6_input_ch = c5_out_ch

        self.stage_out_channels = [-1, -1, -1]

        self.stage_out_channels.append(c2_out_ch)
        self.stage_out_channels.append(c3_out_ch)
        self.stage_out_channels.append(c4_out_ch)
        self.stage_out_channels.append(c5_out_ch)
        self.stage_out_channels.append(c6_out_ch)
        self.stage_out_channels.append(c7_out_ch)

        #[2,3,4,5,6,7] the concat feature map from p2 to p7

        self.attention_modules = []
        self.attention_modules.append(nn.Conv2d(256, 256, 3, 1, 1, bias=False))
        self.attention_modules.append(nn.Conv2d(256, 256, 3, 1, 1, bias=False))
        self.attention_modules.append(nn.Conv2d(256, 256, 3, 1, 1, bias=False))
        self.attention_modules.append(nn.Conv2d(256, 256, 3, 1, 1, bias=False))
        self.attention_modules.append(nn.Conv2d(256, 256, 3, 1, 1, bias=False))
        self.attention_modules = nn.Sequential(*self.attention_modules)

        self.features = []


        self.ct = ContextTexture(self.use_attention)

        self.latlayer = []
        self.smooth_c3 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.smooth_c4 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.smooth_c5 = nn.Conv2d(256,256,kernel_size=3,padding=1)

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
        self.fpn_architecture = fpn_architecture
        for i in range(6):
            self.latlayer.append(nn.Conv2d(self.stage_out_channels[i+3], 256, 1))

        self.latlayer = nn.Sequential(*self.latlayer)
        self.sigmoid = nn.Sigmoid()

    def forward(self, layer_list):
        conv_6 = self.c6(layer_list[-1])
        layer_list.append(conv_6)
        conv_7 = self.c7(layer_list[-1])
        layer_list.append(conv_7)
        latlayer_list = []
        attention_module_list = []
        ret_list = []
        for idx,layer in enumerate(layer_list):
            latlayer  = self.latlayer[idx](layer)
            latlayer_list.append(latlayer)
            if idx!= 5:
                attention_module = self.attention_modules[idx](latlayer)
                attention_module = self.sigmoid(attention_module.mean(1).unsqueeze(1)).expand_as(latlayer)
                attention_module_list.append(attention_module)
            
        for i in range(len(self.fpn_architecture) // 2):
            select_layer_idx = self.fpn_architecture[2*i]
            select_concat_layer_idx = self.fpn_architecture[2*i+1]
            # add smooth channel to 256
            concat_layer = self.ct(latlayer_list[select_concat_layer_idx-2], latlayer_list[select_layer_idx-2], attention_module_list[select_layer_idx-2])
            ret_list.append(concat_layer)
            latlayer_list.append(concat_layer)
        
        ret_list.append(latlayer_list[4])
        ret_list.append(latlayer_list[5])
        ret_list.sort(key= lambda x: x.shape[2], reverse=True)
        ret_list[0] = self.smooth_c3(ret_list[0])
        ret_list[1] = self.smooth_c4(ret_list[1])
        ret_list[2] = self.smooth_c5(ret_list[2])
        return ret_list

    def weight_init(self):
        for layer in self.modules():
            layer.apply(self.weight_init_fn)
 
