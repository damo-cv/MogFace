import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
import torch
from mmdet.ops import (ContextBlock, GeneralizedAttention, build_conv_layer,
                       build_norm_layer)

from mmdet.core import auto_fp16
from ..builder import NECKS


@NECKS.register_module()
class RFFPN(nn.Module):
    """
    Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 use_gap=False,
                 use_gap_all_layer=False,
                 use_deep_gap=False,
                 use_deep_gap_v1=False,
                 gap_with_share_conv=False,
                 use_conv_module=True,
                 gap_use_conv=True,
                 gap_use_sac=False,
                 freeze_gap=False,
                 use_all_layer_ft=False,
                 use_fpn=True,
                 use_smooth_convs=False,
                 weighted_fpn=False,
                 fix_fpn=False):
        super(RFFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.use_gap = use_gap
        self.use_gap_all_layer = use_gap_all_layer
        self.freeze_gap = freeze_gap
        self.use_deep_gap = use_deep_gap
        self.use_deep_gap_v1 = use_deep_gap_v1
        self.gap_with_share_conv = gap_with_share_conv
        self.use_conv_module = use_conv_module
        self.gap_use_sac = gap_use_sac
        self.gap_use_conv = gap_use_conv
        self.use_fpn=use_fpn
        self.use_all_layer_ft = use_all_layer_ft
        self.weighted_fpn = weighted_fpn
        self.fix_fpn = fix_fpn

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.use_smooth_convs = use_smooth_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        if self.use_smooth_convs:
            self.smooth_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        self.smooth_down_convs = nn.ModuleList()
        self.up_ft_weights = nn.ModuleList()
        if self.use_gap:
            self.gap_conv = nn.Conv2d(in_channels[-1], out_channels, 1)
        if self.use_gap_all_layer:
            gap_all_layer_list = []
            for i in range(self.start_level, len(in_channels)):
                if self.gap_use_sac:
                    gap_conv_cfg = dict(type='SAC', use_deform=False)
                    gap_conv = build_conv_layer(
                        gap_conv_cfg,
                        in_channels[i],
                        out_channels,
                        kernel_size=1,
                        padding=0,
                        )
                elif conv_cfg is None or self.gap_use_conv:
                    gap_conv = nn.Conv2d(in_channels[i], out_channels, 1)
                else:
                    gap_conv = build_conv_layer(
                        conv_cfg,
                        in_channels[i],
                        out_channels,
                        kernel_size=1,
                        padding=0,
                        )
                gap_all_layer_list.append(gap_conv)
            self.gap_all_layer = nn.ModuleList(gap_all_layer_list)

        
        for i in range(self.start_level, self.backbone_end_level):
            if self.use_conv_module:
                if self.use_all_layer_ft:
                    down_conv = build_conv_layer(
                        None,
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False)
                    stage_block = [3, 4, 6, 3]
                    l_convs = nn.ModuleList()
                    for _ in range(stage_block[i]):
                        l_conv = ConvModule(
                            in_channels[i],
                            out_channels,
                            1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                            act_cfg=act_cfg,
                            inplace=False)
                        l_convs.append(l_conv)

                if self.use_smooth_convs:
                    smooth_conv = ConvModule(
                        in_channels[i],
                        out_channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                        act_cfg=act_cfg,
                        inplace=False)
                    smooth_down_conv = ConvModule(
                        out_channels * 2,
                        out_channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                        act_cfg=act_cfg,
                        inplace=False)

                    up_ft_weight = torch.nn.Conv2d(
                        out_channels,
                        1,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True)
                    up_ft_weight.weight.data.fill_(0)
                    up_ft_weight.bias.data.fill_(0)
                    self.up_ft_weights.append(up_ft_weight)

                l_conv = ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False)
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
            else:
                l_conv = build_conv_layer(
                    conv_cfg,
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    #norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    )
                fpn_conv = build_conv_layer(
                    conv_cfg,
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    )

            if self.use_all_layer_ft:
                self.down_convs.append(down_conv)
                self.lateral_convs.append(l_convs)
            else:
                self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

            if self.use_smooth_convs:
                self.smooth_convs.append(smooth_conv)
                self.smooth_down_convs.append(smooth_down_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                if self.use_conv_module:
                    extra_fpn_conv = ConvModule(
                        in_channels,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False)
                else:
                    extra_fpn_conv = build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        )
                self.fpn_convs.append(extra_fpn_conv)
        if self.weighted_fpn:
            self.fpn_weights = nn.ModuleList()
            for i in range(self.start_level, self.backbone_end_level):
                fpn_weight = torch.nn.Conv2d(
                    out_channels,
                    1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True)
                fpn_weight.weight.data.fill_(0)
                fpn_weight.bias.data.fill_(0)
                self.fpn_weights.append(fpn_weight)

        #self.init_weights()
        if self.freeze_gap:
            self._freeze_gap()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def _freeze_gap(self):
        if self.use_gap_all_layer:
            for param in self.gap_all_layer.parameters():
                param.requires_grad = False

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # regenerate input feature
        if self.use_all_layer_ft:
            laterals = []
            for i, lateral_conv in enumerate(self.lateral_convs):
                per_lateral_list= []
                tmp_input = inputs[i + self.start_level]
                for tmp_idx,ft in enumerate(tmp_input):
                    per_lateral_list.append(self.down_convs[i](F.interpolate(lateral_conv[tmp_idx](ft), scale_factor=(2,2))))
                    
                laterals.append(torch.stack(per_lateral_list).sum(0))
        else:
            # build laterals
            if self.use_smooth_convs:
                laterals = [
                    lateral_conv(inputs[i + self.start_level])
                    for i, lateral_conv in enumerate(self.lateral_convs)
                ]
                smooth_fts= [
                    smooth_conv(inputs[i + self.start_level])
                    for i, smooth_conv in enumerate(self.smooth_convs)
                ]
            else:
                laterals = [
                    lateral_conv(inputs[i + self.start_level])
                    for i, lateral_conv in enumerate(self.lateral_convs)
                ]

        # build top-down path
        used_backbone_levels = len(laterals)
        if self.use_gap:
            gap =inputs[-1].mean(3,keepdim=True).mean(2,keepdim=True).expand_as(inputs[-1])
            gap = self.gap_conv(gap)
            laterals[-1] = laterals[-1] + gap
           
        if self.use_gap_all_layer:
            gap_list = []
            for i in range(self.start_level, len(inputs)):
                gap = inputs[i].mean(3,keepdim=True).mean(2,keepdim=True).expand_as(inputs[i])
                gap = self.gap_all_layer[i - self.start_level](gap)
                gap_list.append(gap)
            for i in range(len(gap_list)):
                if self.gap_with_share_conv:
                    laterals[i] = laterals[i] + self.gap_share_conv(gap_list[i])
                else:
                    laterals[i] = laterals[i] + gap_list[i]


        if self.use_smooth_convs:
            new_laterals = [None, None, None,]
            for i in range(used_backbone_levels - 1, 0, -1):
                prev_shape = laterals[i - 1].shape[2:]
                upsample_ft = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
                #import pdb;pdb.set_trace()
                #add_weight = torch.sigmoid(self.up_ft_weights[i](upsample_ft))
                add_weight = torch.sigmoid(self.up_ft_weights[i-1](smooth_fts[i-1]))
                laterals[i - 1] +=  upsample_ft 
                new_laterals[i - 1] = laterals[i - 1] + add_weight * smooth_fts[i - 1]
                #laterals[i - 1] +=  add_weight * smooth_fts[i - 1]
                #laterals[i - 1] +=  add_weight * upsample_ft 
                if i == used_backbone_levels - 1:
                    add_weight = torch.sigmoid(self.up_ft_weights[i](smooth_fts[i]))
                    new_laterals[i] = laterals[i] + add_weight * smooth_fts[i]
                    #laterals[i] +=  add_weight * smooth_fts[i]
            laterals = new_laterals


        elif self.use_fpn:
            for i in range(used_backbone_levels - 1, 0, -1):
                # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
                #  it cannot co-exist with `size` in `F.interpolate`.
                if self.weighted_fpn:
                    prev_shape = laterals[i - 1].shape[2:]
                    upsample_ft = F.interpolate(
                        laterals[i], size=prev_shape, **self.upsample_cfg)
                    weight = torch.sigmoid(self.fpn_weights[i](upsample_ft))
                    laterals[i - 1] = laterals[i - 1] + weight * upsample_ft
                    with torch.no_grad():
                        laterals[i - 1] = laterals[i - 1] +  (1 - weight) * upsample_ft

                else:
                    if 'scale_factor' in self.upsample_cfg:
                        laterals[i - 1] += F.interpolate(laterals[i],
                                                         **self.upsample_cfg)
                    else:
                        prev_shape = laterals[i - 1].shape[2:]
                        if self.fix_fpn:
                            with torch.no_grad():
                                upsample_ft = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
                        else:
                            upsample_ft = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
                        laterals[i - 1] += upsample_ft

        # build outputs
        # part 1: from original levels
        if False:
            outs = [
                self.fpn_convs[i](smooth_convs[i]) for i in range(used_backbone_levels)
            ]
        else:
            outs = [
                self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    if False:
                        extra_source = smooth_convs[-1]
                    else:
                        extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                if self.use_all_layer_ft:
                    outs.append(self.fpn_convs[used_backbone_levels](extra_source[-1]))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
