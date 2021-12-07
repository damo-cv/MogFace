# ******************************************************
# Author        : liuyang
# Last modified : 2020-01-13 20:49
# Email         : gxly1314@gmail.com
# Filename      : image_util.py
# Description   : 
# ******************************************************
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy import random as random
import math
import cv2
from utils import bbox_utils

def mog_sse(settings, bbox_labels, sample_bbox_scale):
    if hasattr(settings, '{}_scale_range'.format(settings.main_pyramid_layer)):
        random_num = np.random.random()

        main_scale_range = getattr(settings, '{}_scale_range'.format(settings.main_pyramid_layer))
        main_start_scale = main_scale_range[0]
        main_end_scale = main_scale_range[1]

        aux_scale_range = getattr(settings, '{}_scale_range'.format(settings.auxiliary_pyramid_layer))
        aux_start_scale = aux_scale_range[0]
        aux_end_scale = aux_scale_range[1]

        if random_num < settings.mpl_scale_ratio:
            tmp_scale = random.uniform(main_start_scale, main_end_scale)
            ratio = tmp_scale / sample_bbox_scale
        elif random_num < (settings.mpl_scale_ratio + settings.apl_scale_ratio):
            tmp_scale = random.uniform(aux_start_scale, aux_end_scale)
            ratio = tmp_scale  / sample_bbox_scale
        else:
            tmp_scale = random.uniform(16, main_start_scale)
            ratio =  tmp_scale / sample_bbox_scale
        return ratio

def random_brightness(img, settings):
    prob = np.random.uniform(0, 1)
    if prob < settings.brightness_prob:
        delta = np.random.uniform(-settings.brightness_delta,
                                  settings.brightness_delta) 
        img += delta
    return img


def random_contrast(img, settings):
    prob = np.random.uniform(0, 1)
    if prob < settings.contrast_prob:
        delta = np.random.uniform(settings.contrast_lower_delta,
                                  settings.contrast_upper_delta)
        img *= delta 
    return img


def random_saturation(img, settings):
    prob = np.random.uniform(0, 1)
    if prob < settings.saturation_prob:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 1] *= np.random.uniform(settings.saturation_lower_delta,
                                  settings.saturation_upper_delta) 
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


def random_hue(img, settings):
    prob = np.random.uniform(0, 1)
    if prob < settings.hue_prob:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 0] += random.uniform(-settings.hue_delta, settings.hue_delta)
        img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
        img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


def distort_image(img, settings):
    """
    align
    """
    prob = np.random.uniform(0, 1)
    #prob = np.random.randint(2)
    # Apply different distort order
    if prob > 0.5:
        img = random_brightness(img, settings)
        img = random_contrast(img, settings)
        img = random_saturation(img, settings)
        img = random_hue(img, settings)
    else:
        img = random_brightness(img, settings)
        img = random_saturation(img, settings)
        img = random_hue(img, settings)
        img = random_contrast(img, settings)
    return img


def das_sample_bbox(img, bbox_labels, settings):
    """
    img: arr bgr h,w,3
    bbox_labels: arr [[xmin, ymin, xmax, ymax, label]]
    return: sampled_bbox (list), img, bbox_labels, for img and bbox_labels may be changed
    """
    #import random

    num_gt = len(bbox_labels)
    rand_idx = np.random.randint(0, num_gt) if num_gt != 0 else 0
    height, width, _ = img.shape
    scale_list = settings.das_scale_list
    img_max_size = settings.img_max_size
    interp_methods_list = settings.interp_methods_list
    sample_bbox_width = settings.sample_bbox_width
    sample_bbox_height = settings.sample_bbox_height
    max_trial = settings.max_trial
    
    if num_gt == 0:
        return [], img, bbox_labels
    else:
        xmin = bbox_labels[rand_idx][0]
        ymin = bbox_labels[rand_idx][1]
        xmax = bbox_labels[rand_idx][2]
        ymax = bbox_labels[rand_idx][3]

        wid = xmax - xmin + 1
        hei = ymax - ymin + 1
        sample_bbox_area = wid * hei
        sample_bbox_scale = sample_bbox_area ** 0.5
    
        anchor_idx = len(scale_list) - 1
        distance = 9999999

        for i, anchor_scale in enumerate(scale_list):
            if abs(anchor_scale - sample_bbox_scale) < distance:
                distance = abs(anchor_scale - sample_bbox_scale)
                anchor_idx = i

        target_anchor = random.choice(scale_list[0:min(anchor_idx+1,5)+1])
        if hasattr(settings, 'prob_large_sample'):
            if target_anchor == scale_list[0]:
                if random.random() < settings.prob_large_sample:
                    target_anchor = random.choice(scale_list[1:4])

        if sample_bbox_scale == 0:
            return [], image, bbox_labels
        elif len(scale_list) == 1:
            ratio = float(target_anchor) / sample_bbox_scale
        else:
            ratio = float(target_anchor) / sample_bbox_scale * (2**random.uniform(-1,1))

                
        # constrain the max scale of image

        if settings.use_sse:
            ratio = mog_sse(settings, bbox_labels, sample_bbox_scale)
            #print ('sse time: {:.4f}'.format(end_time - st_time))

        if height * ratio * width * ratio > img_max_size ** 2:
            ratio = (img_max_size ** 2 / (height * width))**0.5

        interp_method = random.choice(interp_methods_list)
        img = cv2.resize(img, None, None, fx=ratio, fy=ratio, interpolation=interp_method)

        w = bbox_labels[:,2] - bbox_labels[:,0] + 1
        h = bbox_labels[:,3] - bbox_labels[:,1] + 1
        bbox_labels[:,0] *= ratio
        bbox_labels[:,1] *= ratio
        bbox_labels[:,2] = bbox_labels[:,0] + w * ratio - 1
        bbox_labels[:,3] = bbox_labels[:,1] + h * ratio - 1

        height, width, _ = img.shape
        # generate sample box
        # TODU: have som bugs bw,  bh should be w,h of choiced bbox after resizing
        xmin = bbox_labels[rand_idx, 0]
        ymin = bbox_labels[rand_idx, 1]
        bw = bbox_labels[rand_idx, 2] - bbox_labels[rand_idx, 0] + 1
        bh = bbox_labels[rand_idx, 3] - bbox_labels[rand_idx, 1] + 1
        sample_boxes = []
        w = sample_bbox_width
        h = sample_bbox_height

        for _ in range(max_trial):
            if w < max(height,width):
                if bw <= w:
                    w_off = random.uniform(xmin + bw - w, xmin)
                else:
                    w_off = random.uniform(xmin, xmin + bw - w)

                if bh <= h:
                    h_off = random.uniform(ymin + bh - h, ymin)
                else:
                    h_off = random.uniform(ymin, ymin + bh -h)
            else:
                w_off = random.uniform(width - w, 0)
                h_off = random.uniform(height - h, 0)

            w_off = math.floor(w_off)
            h_off = math.floor(h_off)

            # convert to integer rect x1,y1,x2,y2
            rect = np.array([int(w_off), int(h_off), int(w_off+w), int(h_off+h)])

            # keep overlap with gt box IF center in sampled patch
            #centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            # mask in all gt boxes that above and to the left of centers
            m1 = (rect[0] <= bbox_labels[:, 0]) * (rect[1] <= bbox_labels[:, 1])
            # mask in all gt boxes that under and to the right of centers
            m2 = (rect[2] >= bbox_labels[:, 2]) * (rect[3] >= bbox_labels[:, 3])
            # mask in that both m1 and m2 are true
            mask = m1 * m2
            
            overlap = bbox_utils.bbox_overlap(bbox_labels[:,:4],rect)
            # have any valid boxes? try again if not
            if not mask.any() and not overlap.max() > 0.7:
                continue
            else:
                sample_boxes.append(rect)

        return sample_boxes, img, bbox_labels

def das_crop_img(img, bbox_labels, sampled_bboxes, settings):
    # no clipping here
    choice_idx = np.random.randint(len(sampled_bboxes))
    sampled_bbox = sampled_bboxes[choice_idx]
    centers = (bbox_labels[:, :2] + bbox_labels[:, 2:4]) / 2.0
    m1 = (sampled_bbox[0] < centers[:, 0]) * (sampled_bbox[1] < centers[:, 1])
    m2 = (sampled_bbox[2] > centers[:, 0]) * (sampled_bbox[3] > centers[:, 1])
    mask = m1 * m2
    # add new regularization
    current_bbox_labels = bbox_labels.copy()
    current_bboxes = current_bbox_labels[mask, :4]
    current_labels = current_bbox_labels[mask, 4]
    
    current_bboxes[:, :2] -= sampled_bbox[:2]
    current_bboxes[:, 2:] -= sampled_bbox[:2]
    #print('boxes', sampled_bbox, current_bboxes)
    current_bboxes = np.clip(current_bboxes, 0, settings.sample_bbox_width - 1)

    img_height, img_width, _ = img.shape

    if sampled_bbox[0] < 0 or sampled_bbox[1] < 0:
        new_img_width = img_width if sampled_bbox[0] >=0 else img_width-sampled_bbox[0]
        new_img_height = img_height if sampled_bbox[1] >=0 else img_height-sampled_bbox[1]
        img_pad = np.zeros((new_img_height,new_img_width,3),dtype=float)
        img_pad[:, :, :] = settings.img_mean
        start_left = 0 if sampled_bbox[0] >=0 else -sampled_bbox[0]
        start_top = 0 if sampled_bbox[1] >=0 else -sampled_bbox[1]
        img_pad[start_top:,start_left:,:] = img

        sampled_bbox_w = sampled_bbox[2] - sampled_bbox[0]
        sampled_bbox_h = sampled_bbox[3] - sampled_bbox[1]

        start_left = sampled_bbox[0] if sampled_bbox[0] >=0 else 0
        start_top = sampled_bbox[1] if sampled_bbox[1] >=0 else 0
        end_right = start_left + sampled_bbox_w
        end_bottom = start_top + sampled_bbox_h
        current_img = img_pad[start_top:end_bottom,start_left:end_right,:].copy()

        current_img_h, current_img_w, _ = current_img.shape

        if current_img_h < settings.sample_bbox_height or current_img_w < \
                           settings.sample_bbox_width:
            img_pad_temp = np.zeros((settings.sample_bbox_height, settings.sample_bbox_width,3))
            img_pad_temp[:, :, :] = settings.img_mean
            img_pad_temp[0:current_img_h,0:current_img_w,:] = \
                    img_pad[start_top:end_bottom,start_left:end_right,:].copy()
            current_img = img_pad_temp.copy()
     
        bbox_labels = np.concatenate([current_bboxes, current_labels[:,np.newaxis]], axis=1)

        return current_img, bbox_labels 

    current_img = img[sampled_bbox[1]:sampled_bbox[3],sampled_bbox[0]:sampled_bbox[2],:].copy()


    current_img_h, current_img_w, _ = current_img.shape

    if current_img_h < settings.sample_bbox_height or current_img_w < \
                       settings.sample_bbox_height:
        img_pad_temp = np.zeros((settings.sample_bbox_height, settings.sample_bbox_width,3))
        img_pad_temp[:, :, :] = settings.img_mean
        img_pad_temp[0:current_img_h,0:current_img_w,:] = img[sampled_bbox[1]:sampled_bbox[3],sampled_bbox[0]:sampled_bbox[2],:].copy()
        current_img = img_pad_temp.copy()

    bbox_labels = np.concatenate([current_bboxes, current_labels[:,np.newaxis]], axis=1)
     
    return current_img, bbox_labels

def random_mirror(img, bbox_labels):
    #mirror = int(np.random.uniform(0, 2))
    if bbox_labels.shape[0] == 0:
        return img, bbox_labels
    mirror = np.random.randint(2)
    if mirror == 1:
        img = img[:, ::-1, :]
        _, width, _ = img.shape
        tmp = bbox_labels.copy()[:, 0]
        bbox_labels[:, 0] = width - bbox_labels[:, 2] - 1
        bbox_labels[:, 2] = width - tmp - 1

    return img, bbox_labels

def resize_img(img, bbox_labels, settings, resize_height=None, resize_width=None, resize_ratio=None):
    img_h, img_w, _ = img.shape
    #interp_indx = np.random.randint(0, len(settings.interp_methods_list))
    interp_method = np.random.choice(settings.interp_methods_list).item()
    if resize_height is None:
        img = cv2.resize(img, (settings.resize_width, settings.resize_height),
                         interpolation=interp_method)
    else:
        ratio_w = resize_width / img_w
        ratio_h = resize_height / img_h
        ratio = min(ratio_w, ratio_h)
        img = cv2.resize(img, None, fx=ratio, fy=ratio,
                         interpolation=interp_method)
    if bbox_labels.shape[0] == 0:
        return img, bbox_labels
    else:
        if resize_height is None:
            ratio_w = settings.resize_width / img_w
            ratio_h = settings.resize_height / img_h
        else:
            ratio_w = ratio
            ratio_h = ratio

        w = bbox_labels[:, 2] - bbox_labels[:, 0] + 1
        h = bbox_labels[:, 3] - bbox_labels[:, 1] + 1

        bbox_labels[:, 0] *= ratio_w 
        bbox_labels[:, 1] *= ratio_h
        bbox_labels[:, 2] = bbox_labels[:, 0] + w * ratio_w - 1
        bbox_labels[:, 3] = bbox_labels[:, 1] + h * ratio_h - 1

        return img, bbox_labels

def normalize_img(img, settings):
    """
    TODO: check the detail
    """
    img = img.astype(np.float32)
    img -= settings.img_mean
    img /= settings.img_std
    if settings.normalize_pixel:
        img /= 255
    if settings.use_rgb:
        img = img[:,:,::-1].copy()
    return img

def filter_face(img, bbox_labels, settings):
    del_idx = []
    if bbox_labels.shape[0] == 0:
        return img, bbox_labels
    for i in range(bbox_labels.shape[0]):
        width = bbox_labels[i][2] - bbox_labels[i][0] + 1
        height = bbox_labels[i][3] - bbox_labels[i][1] + 1
        if settings.max_face_size is not None:
            if math.sqrt(width * height) > settings.max_face_size:
                del_idx.append(i)
        if math.sqrt(width * height) < settings.min_face_size:
            del_idx.append(i)
    if len(del_idx) != len(bbox_labels):
        bbox_labels = np.delete(bbox_labels, del_idx, 0)

    return img, bbox_labels
