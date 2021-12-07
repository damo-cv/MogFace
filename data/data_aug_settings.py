# ******************************************************
# Author        : liuyang
# Last modified : 2020-01-13 20:55
# Email         : gxly1314@gmail.com
# Filename      : data_aug_settings.py
# Description   : 
# ******************************************************
import numpy as np
import cv2

import sys
from core.workspace import register

@register
class DataAugSettings(object):
    def __init__(self, das_scale_list=[16, 32, 64, 128, 256, 512], \
                p2_scale_range=None, p3_scale_range=None, p4_scale_range=None, \
                p5_scale_range=None, p6_scale_range=None, p7_scale_range=None, \
                img_resize_range=None, main_pyramid_layer=None, auxiliary_pyramid_layer=None, \
                resize_height=640, resize_width=640, mpl_scale_ratio=None, apl_scale_ratio=None, \
                sample_bbox_width=640, sample_bbox_height=640, max_scale=None, max_face_size=None, \
                use_mst=False, min_face_size=8, use_sse=False, use_rsc=False, prob_large_sample=0.0, img_max_size=12000):
        # data aug
        self.apply_distort = True
        self.resize_height = resize_height
        self.resize_width = resize_width
        
        # distort
        self.brightness_prob = 0.5
        self.brightness_delta = 32

        self.contrast_prob = 0.5
        self.contrast_lower_delta = 0.5
        self.contrast_upper_delta = 1.5

        self.saturation_prob = 0.5
        self.saturation_lower_delta = 0.5
        self.saturation_upper_delta = 1.5

        self.hue_prob = 0.5
        self.hue_delta = 18

        self.das_prob = 1
        self.das_scale_list = das_scale_list
        self.img_max_size = img_max_size
        self.interp_methods_list = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, \
                                cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        self.sample_bbox_width = sample_bbox_width
        self.sample_bbox_height = sample_bbox_height
        self.max_trial = 50

        self.min_face_size = min_face_size
        self.max_face_size = max_face_size
        self.prob_large_sample = prob_large_sample

        self.p2_scale_range = p2_scale_range
        self.p3_scale_range = p3_scale_range
        self.p4_scale_range = p4_scale_range
        self.p5_scale_range = p5_scale_range
        self.p6_scale_range = p6_scale_range
        self.p7_scale_range = p7_scale_range

        self.mpl_scale_ratio = mpl_scale_ratio
        self.apl_scale_ratio = apl_scale_ratio

        self.main_pyramid_layer = main_pyramid_layer
        self.auxiliary_pyramid_layer = auxiliary_pyramid_layer
        self.max_scale = max_scale
        self.use_sse = use_sse

        self.img_resize_range = img_resize_range
        self.use_mst = use_mst
        self.use_rsc = use_rsc
