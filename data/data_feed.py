# ******************************************************
# Author        : liuyang
# Last modified : 2020-01-13 20:47
# Email         : gxly1314@gmail.com
# Filename      : data_feed.py
# Description   : 
# ******************************************************

from __future__ import absolute_import
import os
import math
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from core.workspace import register

@register
class TrainFeed(data.Dataset):
    __shared__ = ['out_bbox_anchor']
    __inject__ = ['dataset', 'collate_fn']
    def __init__(self, dataset, collate_fn=None, dataset_name='WiderFace', add_dsfd_label=False, \
                out_bbox_anchor=False):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.dataset_name = dataset_name
        self.add_dsfd_label = add_dsfd_label
        self.out_bbox_anchor = out_bbox_anchor

    def __getitem__(self, index):
        if self.add_dsfd_label:
            img, bbox_labels, dsfd_bbox_labels = self.dataset[index]
            return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(bbox_labels), dsfd_bbox_labels
        elif self.out_bbox_anchor:
            img, encode_bbox_labels, anchors, bbox_labels = self.dataset[index]
            return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(encode_bbox_labels), \
                   torch.from_numpy(anchors), bbox_labels 
        else:
            img, bbox_labels = self.dataset[index]
            return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(bbox_labels)


    def __len__(self):
        return len(self.dataset)

@register 
class TrainCollate(object):
    def __init__(self):
        pass
    def __call__(self, batch):
        bbox_labels_list = []
        img_list = []
        for sample in batch:
            img_list.append(sample[0])
            bbox_labels_list.append(torch.FloatTensor(sample[1]))
        return torch.stack(img_list, 0), bbox_labels_list


@register 
class TrainTransformerCollate(object):
    def __init__(self):
        pass
    def __call__(self, batch):
        bbox_labels_list = []
        img_list = []
        mask_list = []
        for sample in batch:
            img_list.append(sample[0])
            bbox_labels_list.append(sample[1])
            mask_list.append(torch.FloatTensor(sample[2]))
        return torch.stack(img_list, 0), bbox_labels_list, torch.stack(mask_list, 0)

@register 
class BBoxAnchorTrainCollate(object):
    def __init__(self):
        pass
    def __call__(self, batch):
        img_list = []
        encode_bbox_labels_list = []
        bbox_labels_list = []

        for sample in batch:
            img_list.append(sample[0])
            encode_bbox_labels_list.append(sample[1])
            bbox_labels_list.append(torch.FloatTensor(sample[3]))

        return torch.stack(img_list, 0), torch.stack(encode_bbox_labels_list, 0), \
               sample[2], bbox_labels_list



if __name__ == '__main__': 
    from data.datasets_utils import widerface
    from tools.visualize import draw_bboxes
    from data import preprocess
    import yaml
    with open('data_loader.yml') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    cfg = load_config('data_loader.yml')
    dataset = widerface.WiderFace()
    import pdb;pdb.set_trace()
    length = len(dataset)
    # TODO rebuild preprocess op and visualize img
    pre_process = preprocess.Preprocess()
    detection = Detection(dataset, pre_process)
    random_sample_arr = np.random.randint(0, length, 100)
    os.system('rm -rf tmp_img/*')
    for i in range(random_sample_arr.shape[0]):
        img, bbox_labels = detection.pull_item(random_sample_arr[i])
        draw_bboxes(np.array(img).transpose([1,2,0]), np.array(bbox_labels)[:, :4], output_dir='./tmp_img')


