# ******************************************************
# Author        : liuyang
# Last modified : 2020-01-13 20:48
# Email         : gxly1314@gmail.com
# Filename      : preprocess.py
# Description   : 
# ******************************************************
from __future__ import absolute_import
from .transform import image_util
import numpy as np
from core.workspace import register
from .anchors_opr import *

@register
class BasePreprocess(object):
    # TODO add __share__for img_mean and img_std
    __inject__ = ['data_aug_settings']
    #__shared__ = ['img_mean', 'img_std', 'normalize_pixel', 'use_rgb']
    def __init__(self, data_aug_settings, img_mean=[104., 117., 123.], img_std=[1., 1., 1.], normalize_pixel=False, use_rgb=False):
        self.settings = data_aug_settings
        # data aug
        self.settings.use_rgb = use_rgb
        if self.settings.use_rgb:
            self.settings.img_mean = (np.array(img_mean).astype('float32') * 255)[::-1]
            self.settings.img_std = (np.array(img_std).astype('float32') * 255)[::-1]
        else:
            self.settings.img_mean = np.array(img_mean).astype('float32') 
            self.settings.img_std = np.array(img_std).astype('float32') 
        self.settings.normalize_pixel = normalize_pixel
        
    def __call__(self, img, bbox_labels=None, phase='training', data_aug_settings=None):
        img_height, img_width, _ = img.shape
        if phase == 'training':
            if self.settings.use_mst:
                self.settings.resize_height  = data_aug_settings.resize_height
                self.settings.resize_width = data_aug_settings.resize_width
                self.settings.sample_bbox_width = data_aug_settings.sample_bbox_width
                self.settings.sample_bbox_height = data_aug_settings.sample_bbox_height
            if self.settings.apply_distort:
                img = image_util.distort_image(img, self.settings)

            prob = np.random.uniform(0., 1.)
            # sse
            if prob < self.settings.das_prob:
                if bbox_labels.shape[0] != 0:
                    sampled_bboxes, img, bbox_labels = image_util.das_sample_bbox(
                        img, bbox_labels, self.settings)
                    if len(sampled_bboxes) > 0:
                        img, bbox_labels= image_util.das_crop_img(
                            img, bbox_labels, sampled_bboxes, self.settings)
                        #print ('len: ', len(sampled_bboxes), bbox_labels.shape)
            #default mirror
            img, bbox_labels = self.preprocess_1(img, bbox_labels)

            return img, bbox_labels
        else:
            img = image_util.normalize_img(img, self.settings)

            return img

    def preprocess_1(self, img, bbox_labels):
        img, bbox_labels = image_util.random_mirror(img, bbox_labels)
        img, bbox_labels = image_util.resize_img(img, bbox_labels, self.settings)
        img = image_util.normalize_img(img, self.settings)
        img, bbox_labels = image_util.filter_face(img, bbox_labels, self.settings)
        return img, bbox_labels
