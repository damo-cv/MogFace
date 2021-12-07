# ******************************************************
# Author        : liuyang
# Last modified : 2020-01-13 20:46
# Email         : gxly1314@gmail.com
# Filename      : anchor_target.py
# Description   : 
# ******************************************************
import numpy as np 
from core.workspace import register
from . import anchor_utils
from utils import bbox_utils

@register
class AnchorTarget(object):
    def __init__(self, rpn_positive_overlap=0.35, \
                 rpn_negative_overlap=0.35, \
                 gt_match_maxoverlap_anchor=True, \
                 nomalized_compute_overlap=False):
        self.rpn_positive_overlap = rpn_positive_overlap
        self.rpn_negative_overlap = rpn_negative_overlap
        self.gt_match_maxoverlap_anchor = gt_match_maxoverlap_anchor
        self.nomalized_compute_overlap = nomalized_compute_overlap

    def __call__(self, anchors, bbox_labels):
        num_anchors = anchors.shape[0]
        labels = np.empty((num_anchors,1), dtype='float32')
        overlaps = bbox_utils.bbox_overlap(anchors, bbox_labels[:,:4], self.nomalized_compute_overlap)
        # from anchor perspective [num_anchors, 1]
        argmax_overlaps_for_anchor = overlaps.argmax(axis=1)
        max_overlaps_for_anchor = overlaps[[True] * overlaps.shape[0], argmax_overlaps_for_anchor]

        # from gt perspective
        argmax_overlaps_for_gt = overlaps.argmax(axis=0)
        #max_overlaps_for_gt = ovrlaps[argmax_overlaps_for_gt, :]

        if self.gt_match_maxoverlap_anchor:
            for i in range(argmax_overlaps_for_gt.shape[0]):
                argmax_overlaps_for_anchor[argmax_overlaps_for_gt[i]] = i
                # ensure anchors that have the best_overlap with gt to be trained
                max_overlaps_for_anchor[argmax_overlaps_for_gt[i]] = 1

        gt_boxes = bbox_labels[:, :4].copy()
        anchor_matched_gt = gt_boxes[argmax_overlaps_for_anchor]

        labels[max_overlaps_for_anchor < self.rpn_negative_overlap] = 0
        labels[max_overlaps_for_anchor >= self.rpn_positive_overlap] = 1
        
        bbox_targets = anchor_utils.encode(anchor_utils.transform_anchor(anchors), \
                                           anchor_matched_gt)
        targets =  np.concatenate([bbox_targets, labels], axis=1) 
        return targets


