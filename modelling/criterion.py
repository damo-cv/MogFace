# ******************************************************
# Author        : liuyang
# Last modified : 2020-01-13 20:43
# Email         : gxly1314@gmail.com
# Filename      : criterion.py
# Description   : 
# ******************************************************
import torch.nn as nn
import torch
from core.workspace import register
from data.anchors_opr import anchor_utils
import torch.nn.functional as F
from utils import bbox_utils

@register
class CriterionFPBranch(nn.Module):
    __inject__ = ['cls_loss_fn']
    def __init__(self, cls_loss_fn=None, cls_loss_weight=1
                 ):
        super(CriterionFPBranch, self).__init__()
        self.cls_loss_fn = cls_loss_fn

        self.cls_loss_weight = cls_loss_weight
        
    def forward(self, conf, fp_label):
        labels = fp_label
        cls_loss = self.cls_loss_fn(conf, labels)
        return self.cls_loss_weight * cls_loss 

@register
class Criterion(nn.Module):
    __inject__ = ['cls_loss_fn', 'loc_loss_fn']
    __shared__ = ['fp_th']
    def __init__(self, cls_loss_fn=None, loc_loss_fn=None, \
                 cls_loss_weight=1, loc_loss_weight=2, fp_th=0.5, \
                 out_fp_label=False 
                 ):
        super(Criterion, self).__init__()
        self.cls_loss_fn = cls_loss_fn
        self.loc_loss_fn = loc_loss_fn

        self.cls_loss_weight = cls_loss_weight
        self.loc_loss_weight = loc_loss_weight
        
        self.out_fp_label = out_fp_label
        if self.out_fp_label:
            self.fp_th = fp_th

    def forward(self, conf, loc, bbox_targets):
        labels = bbox_targets[:, :, -1].unsqueeze(2)
        if self.out_fp_label:
            with torch.no_grad():
                fp_label = self.generate_fp_label(conf, labels)
        cls_loss = self.cls_loss_fn(conf, labels)
        loc_loss = self.loc_loss_fn(loc, bbox_targets)
        total_loss = self.cls_loss_weight * cls_loss + self.loc_loss_weight * loc_loss
        if self.out_fp_label:
            return self.cls_loss_weight * cls_loss, self.loc_loss_weight * loc_loss, total_loss, fp_label
        else:
            return self.cls_loss_weight * cls_loss, self.loc_loss_weight * loc_loss, total_loss

    def generate_fp_label(self, conf, labels):
        sigmoid_conf = conf.sigmoid()
        # tp_label = 1, fp_label = 0, remaining anchor = -1
        mask_tp_anchor = (sigmoid_conf > self.fp_th) * labels
        mask_fp_anchor = (sigmoid_conf > self.fp_th) * (1 - labels)

        mask_tp_anchor[mask_tp_anchor == 0] = -1
        mask_tp_anchor[mask_fp_anchor == 1] = 0

        return mask_tp_anchor


@register
class MogCriterion(nn.Module):
    __inject__ = ['cls_loss_fn', 'loc_loss_fn']
    def __init__(self, cls_loss_fn=None, loc_loss_fn=None, \
                 cls_loss_weight=1, loc_loss_weight=2, \
                 ):
        super(MogCriterion, self).__init__()
        self.cls_loss_fn = cls_loss_fn
        self.loc_loss_fn = loc_loss_fn

        self.cls_loss_weight = cls_loss_weight
        self.loc_loss_weight = loc_loss_weight
        
    def forward(self, conf, loc, bbox_targets, anchors, bbox_labels_list):
        num_batch = conf.shape[0]
        labels = bbox_targets[:, :, -1]
        pos_labels = labels > 0
        transformed_anchors = anchor_utils.transform_anchor_opr(anchors)


        with torch.no_grad():
            for batch_idx in range(num_batch):
                pos_anchor_idx = bbox_targets[batch_idx,:,-1] == 1
                num_anchor = conf[batch_idx].shape[0]
                num_gt = bbox_labels_list[batch_idx].shape[0]

                _, sort_idx = F.sigmoid(conf[batch_idx][pos_anchor_idx]).sort(0)
                ignore_ratio = 0.0

                ig_idx = sort_idx[int(len(sort_idx) * ignore_ratio):, :] 
                mask_ig_bbox_target = bbox_targets[batch_idx][pos_anchor_idx][ig_idx].squeeze(1)[:,:4]

                decoded_boxes = anchor_utils.decode(mask_ig_bbox_target, transformed_anchors[pos_anchor_idx][ig_idx].squeeze(1))
                all_decoded_boxes = anchor_utils.decode(loc[batch_idx, :], transformed_anchors)
                p2_decoded_boxes = anchor_utils.decode(bbox_targets[batch_idx][:25600][pos_anchor_idx[:25600]].squeeze(1)[:,:4], transformed_anchors[:25600][pos_anchor_idx[:25600]].squeeze(1))
                p3_decoded_boxes = anchor_utils.decode(bbox_targets[batch_idx][25600:32000][pos_anchor_idx[25600:32000]].squeeze(1)[:,:4], transformed_anchors[25600:32000][pos_anchor_idx[25600:32000]].squeeze(1))
                p4_decoded_boxes = anchor_utils.decode(bbox_targets[batch_idx][32000:33600][pos_anchor_idx[32000:33600]].squeeze(1)[:,:4], transformed_anchors[32000:33600][pos_anchor_idx[32000:33600]].squeeze(1))
                p5_decoded_boxes = anchor_utils.decode(bbox_targets[batch_idx][33600:34000][pos_anchor_idx[33600:34000]].squeeze(1)[:,:4], transformed_anchors[33600:34000][pos_anchor_idx[33600:34000]].squeeze(1))
                p6_decoded_boxes = anchor_utils.decode(bbox_targets[batch_idx][34000:34100][pos_anchor_idx[34000:34100]].squeeze(1)[:,:4], transformed_anchors[34000:34100][pos_anchor_idx[34000:34100]].squeeze(1))
                p7_decoded_boxes = anchor_utils.decode(bbox_targets[batch_idx][34100:][pos_anchor_idx[34100:]].squeeze(1)[:,:4], transformed_anchors[34100:][pos_anchor_idx[34100:]].squeeze(1))

                uniqe_gt_bboxes = torch.unique(decoded_boxes, dim=0)
                unique_gt_bboxes_list = []
                p2_unique_gt_bboxes = torch.unique(p2_decoded_boxes, dim=0, return_counts=True)
                p3_unique_gt_bboxes = torch.unique(p3_decoded_boxes, dim=0, return_counts=True)
                p4_unique_gt_bboxes = torch.unique(p4_decoded_boxes, dim=0, return_counts=True)
                p5_unique_gt_bboxes = torch.unique(p5_decoded_boxes, dim=0, return_counts=True)
                p6_unique_gt_bboxes = torch.unique(p6_decoded_boxes, dim=0, return_counts=True)
                p7_unique_gt_bboxes = torch.unique(p7_decoded_boxes, dim=0, return_counts=True)

                unique_gt_bboxes_list.append(p2_unique_gt_bboxes)
                unique_gt_bboxes_list.append(p3_unique_gt_bboxes)
                unique_gt_bboxes_list.append(p4_unique_gt_bboxes)

                unique_gt_bboxes_list.append(p5_unique_gt_bboxes)
                unique_gt_bboxes_list.append(p6_unique_gt_bboxes)
                unique_gt_bboxes_list.append(p7_unique_gt_bboxes)
                gt_bboxes = (bbox_labels_list[batch_idx]).clone()
                relabel_idx_list = []

                #print ('gt_bboxes: ', uniqe_gt_bboxes)
                start_idx = -25600
                end_idx = 0
                tmp_w = 160
                for layer_idx, gt_bboxes in enumerate(unique_gt_bboxes_list):
                    if layer_idx > 4:
                        continue
                    if gt_bboxes[0].shape[0] == 0:
                        continue
                    max_num = gt_bboxes[1].max()
                    pure_gt_bboxes = gt_bboxes[0]
                    start_idx += tmp_w ** 2
                    end_idx += tmp_w ** 2
                    tmp_w /= 2
                    start_idx = int(start_idx)
                    end_idx = int(end_idx)
                    for tmp_idx, gt_bbox in enumerate(pure_gt_bboxes):
                        num_matched_anchor = gt_bboxes[1][tmp_idx]
                        if num_matched_anchor == max_num:
                            continue
                        scale = torch.sqrt(torch.abs((gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1)))
                        top_k = max_num
                        l2_distance, candidate_idx = anchor_utils.get_k_center_anchor(gt_bbox, transformed_anchors[start_idx:end_idx,:],top_k)
                        top_k_2 = max_num - num_matched_anchor
                        relabel_idx = candidate_idx[F.sigmoid(conf[batch_idx,start_idx:end_idx,:][candidate_idx]).sort(0)[1][-top_k_2:]].squeeze(1)
                        relabel_idx +=  start_idx
                        relabel_idx = relabel_idx[labels[batch_idx, relabel_idx] != 1] 
                        encode_loc = anchor_utils.encode_opr(transformed_anchors[relabel_idx], gt_bbox.unsqueeze(0))
                        if relabel_idx.shape[0] == 0:
                            continue
                        labels[batch_idx, relabel_idx] = 1
                        
                        tmp = bbox_targets[batch_idx]
                        tmp_1 = tmp[relabel_idx]
                        tmp_1[:,:4] = encode_loc
                        tmp_1[:,4] = 1
                        tmp[relabel_idx] = tmp_1
                        bbox_targets[batch_idx] = tmp

        cls_loss = self.cls_loss_fn(conf, labels.unsqueeze(2))
        loc_loss = self.loc_loss_fn(loc, bbox_targets)
        total_loss = self.cls_loss_weight * cls_loss + self.loc_loss_weight * loc_loss

        return self.cls_loss_weight * cls_loss, self.loc_loss_weight * loc_loss, total_loss

