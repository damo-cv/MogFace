# -*- coding: utf-8 -*-
import torch
from data import face
import math
import numpy as np

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - (boxes[:, 2:] - 1)/2,     # xmin, ymin
                     boxes[:, :2] + (boxes[:, 2:] - 1)/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2]), 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    if A*B*2 / 1024 / 1024 * 4 > 1000:
        print("Warning! Memory is:", A*B*2 / 1024 / 1024 * 4, "MB")
        box_a_cpu = box_a.cpu()
        box_b_cpu = box_b.cpu()
        max_xy_cpu = torch.min(box_a_cpu[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b_cpu[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy_cpu = torch.max(box_a_cpu[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b_cpu[:, :2].unsqueeze(0).expand(A, B, 2))
        max_xy_cpu[:,:,0] = max_xy_cpu[:,:,0] - min_xy_cpu[:,:,0] 
        max_xy_cpu[:,:,1] = max_xy_cpu[:,:,1] - min_xy_cpu[:,:,1]
        max_xy_cpu.clamp_(min=0)
        res_cpu = max_xy_cpu[:, :, 0] * max_xy_cpu[:, :, 1]
        res = res_cpu

    else:
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        max_xy[:,:,0] = max_xy[:,:,0] - min_xy[:,:,0] 
        max_xy[:,:,1] = max_xy[:,:,1] - min_xy[:,:,1]
        

        max_xy.clamp_(min=0)
        res = max_xy[:, :, 0] * max_xy[:, :, 1]
    return res

def batch_bbox_overlap(rois, gt_boxes):
    '''
    rois [batch_size, num_anchor, 5[batch_idx. decoded_proposal]]
    gt_boxes [batch_size, num_anchor, 5[loc, labels]]
    use for proposal_target_layer()
    return overlaps [batch_size,N,K]
    '''
    batch_size = rois.shape[0]
    N = rois.shape[1]
    K = gt_boxes.shape[1]

    if rois.shape[2] == 4:
        rois = rois[:,:,:4]#.continuous()
    else:
        rois = rois[:,:,1:5]#.continuous()
    
    gt_boxes_w = gt_boxes[:,:,2] - gt_boxes[:,:,0]
    gt_boxes_h = gt_boxes[:,:,3] - gt_boxes[:,:,1]
    gt_boxes_area = (gt_boxes_w * gt_boxes_h).view(batch_size, 1, K)

    rois_w = rois[:,:,2] - rois[:,:,0]
    rois_h = rois[:,:,3] - rois[:,:,1]
    rois_area = (rois_w * rois_h).view(batch_size, N, 1)

    rois = rois.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
    gt_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

    gt_area_zero = (gt_boxes_w == 0) & (gt_boxes_h == 0)
    rois_area_zero = (rois_w == 0) & (rois_h == 0)
    iw = torch.min(rois[:,:,:,2], gt_boxes[:,:,:,2]) - \
            torch.max(rois[:,:,:,0],gt_boxes[:,:,:,0])
    iw[iw < 0] = 0

    ih = torch.min(rois[:,:,:,3], gt_boxes[:,:,:,3]) - \
            torch.max(rois[:,:,:,1],gt_boxes[:,:,:,1])
    ih[ih < 0] = 0

    union = rois_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / union

    #overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size,N,K), -1)
    #overlaps.masked_fill_(rois_area_zero.view(batch_size, N, 1).expand(batch_size,N,K), -1)

    #import pdb; pdb.set_trace()
    return overlaps


def IoU(box_a, box_b):
    """Compute the IoU of two sets of boxes.  
    E.g.:
        A ∩ B / A  = A ∩ B / area(A)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_objects,4]
    Return:
        IoU: (tensor) Shape: [num_objects]
    """
    inter_xmin = torch.max(box_a[:, 0], box_b[:, 0])
    inter_ymin = torch.max(box_a[:, 1], box_b[:, 1])
    inter_xmax = torch.min(box_a[:, 2], box_b[:, 2])
    inter_ymax = torch.min(box_a[:, 3], box_b[:, 3])
    Iw = torch.clamp(inter_xmax - inter_xmin , min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin , min=0)  
    I = Iw * Ih
    box_a_area = (box_a[:, 2] - box_a[:, 0] ) * (box_a[:, 3] - box_a[:, 1])
    box_b_area = (box_b[:, 2] - box_b[:, 0] ) * (box_b[:, 3] - box_b[:, 1] )
    union = box_a_area + box_b_area - I
    #add_one = True
    #if add_one:
    #  inf_idx = (I == 0)
    #  I[inf_idx] = 1
    #  union[inf_idx] = union[inf_idx] + 1
    return I / union


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    #torch.cuda.empty_cache()
    if not inter.is_cuda:
        box_a_cpu = box_a.cpu()
        box_b_cpu = box_b.cpu()
        area_a_cpu = ((box_a_cpu[:, 2]-box_a_cpu[:, 0]) *
              (box_a_cpu[:, 3]-box_a_cpu[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b_cpu = ((box_b_cpu[:, 2]-box_b_cpu[:, 0]) *
              (box_b_cpu[:, 3]-box_b_cpu[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union_cpu = area_a_cpu + area_b_cpu - inter.cpu()
        return inter / union_cpu
    else:
        area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter

        return inter / union  # [A,B]


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx, or_img_shape):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors),
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    if not best_truth_overlap.is_cuda:
        best_prior_overlap = best_prior_overlap.cuda()
        best_prior_idx= best_prior_idx.cuda()
        best_truth_overlap = best_truth_overlap.cuda()
        best_truth_idx = best_truth_idx.cuda()
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    #best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    #for j in range(best_prior_idx.size(0)):
    #    best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances, or_img_shape)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn

    conf_t[idx] = conf  # [num_priors] top class label for each prior



def matchNoBipartite(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    # print('****************************************')
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    if not best_truth_overlap.is_cuda:
        
        best_truth_overlap = best_truth_overlap.cuda()
        best_truth_idx = best_truth_idx.cuda()
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)


    # for i in range(truths.shape[0]):
    #     meet_cond_indx = np.where(best_truth_overlap[np.where(best_truth_idx==i)[0]]>=threshold)[0]
    #     pick_num = meet_cond_indx.shape[0]
    #     print('Before anchor matching, topk is ',pick_num)

    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background

    # above_thres = np.where(best_truth_overlap >= threshold)
    # print('above_thres : ',above_thres)
    # print('size of above_thres : ',above_thres[0].shape[0])

    # print('labels[best_truth_idx] : ',labels[best_truth_idx][0:100])
    # print('v1 --- gt_num : ',truths.shape[0])
    # print('v1 --- total conf : ',np.where(conf>0)[0].shape[0])

    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def MultiPropertymatchNoBipartite(threshold, truths, priors, variances, labels, loc_t, conf_t, idx, 
        blur_prop, exp_prop, occ_prop, pose_prop, blur_t, exp_t, occ_t, pose_t):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    # print('****************************************')
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    if not best_truth_overlap.is_cuda:
        
        best_truth_overlap = best_truth_overlap.cuda()
        best_truth_idx = best_truth_idx.cuda()
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)


    # for i in range(truths.shape[0]):
    #     meet_cond_indx = np.where(best_truth_overlap[np.where(best_truth_idx==i)[0]]>=threshold)[0]
    #     pick_num = meet_cond_indx.shape[0]
    #     print('Before anchor matching, topk is ',pick_num)

    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background

    # above_thres = np.where(best_truth_overlap >= threshold)
    # print('above_thres : ',above_thres)
    # print('size of above_thres : ',above_thres[0].shape[0])

    # print('labels[best_truth_idx] : ',labels[best_truth_idx][0:100])
    # print('v1 --- gt_num : ',truths.shape[0])
    # print('v1 --- total conf : ',np.where(conf>0)[0].shape[0])

    blur = blur_prop[best_truth_idx]
    exp = exp_prop[best_truth_idx]
    occ = occ_prop[best_truth_idx]
    pose = pose_prop[best_truth_idx]

    blur_t[idx] = blur
    exp_t[idx] = exp
    occ_t[idx] = occ
    pose_t[idx] = pose

    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior



def MultiPropertyPoseQualitymatchNoBipartite(threshold, truths, priors, variances, labels, loc_t, conf_t, idx, 
        pose_x_prop, pose_y_prop, pose_z_prop, quality_prop, pose_x_t, pose_y_t, pose_z_t, quality_t):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    # print('****************************************')
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    if not best_truth_overlap.is_cuda:
        
        best_truth_overlap = best_truth_overlap.cuda()
        best_truth_idx = best_truth_idx.cuda()
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)


    # for i in range(truths.shape[0]):
    #     meet_cond_indx = np.where(best_truth_overlap[np.where(best_truth_idx==i)[0]]>=threshold)[0]
    #     pick_num = meet_cond_indx.shape[0]
    #     print('Before anchor matching, topk is ',pick_num)

    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background

    # above_thres = np.where(best_truth_overlap >= threshold)
    # print('above_thres : ',above_thres)
    # print('size of above_thres : ',above_thres[0].shape[0])

    # print('labels[best_truth_idx] : ',labels[best_truth_idx][0:100])
    # print('v1 --- gt_num : ',truths.shape[0])
    # print('v1 --- total conf : ',np.where(conf>0)[0].shape[0])

    quality = quality_prop[best_truth_idx]
    pose_x = pose_x_prop[best_truth_idx]
    pose_y = pose_y_prop[best_truth_idx]
    pose_z = pose_z_prop[best_truth_idx]

    quality_t[idx] = quality.unsqueeze(1)
    pose_x_t[idx] = pose_x.unsqueeze(1)
    pose_y_t[idx] = pose_y.unsqueeze(1)
    pose_z_t[idx] = pose_z.unsqueeze(1)

    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior



def matchNoBipartiteV1(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    # print('****************************************')
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    if not best_truth_overlap.is_cuda:
        
        best_truth_overlap = best_truth_overlap.cuda()
        best_truth_idx = best_truth_idx.cuda()
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx]          # Shape: [num_priors]

    for i in range(truths.shape[0]):
        first_idx = np.where(best_truth_idx==i)[0]
        second_idx = np.where(best_truth_overlap[first_idx]>=threshold)[0]
        pick_num = second_idx.shape[0]
        # print('Before anchor matching, topk is ',pick_num)
        # print('gt_box location : ',truths[i]*640.0)

        sorted_idx = first_idx[second_idx]
        # print('sorted_idx : ',sorted_idx)
        conf[first_idx[second_idx]] = 1  # label as background
    
    
    

    # above_thres = np.where(best_truth_overlap >= threshold)
    # print('above_thres : ',above_thres)
    # print('size of above_thres : ',above_thres[0].shape[0])
    # print('labels[best_truth_idx] : ',labels[best_truth_idx][0:100])
    # print('v1_temp --- gt_num : ',truths.shape[0])
    # print('v1_temp --- total conf : ',np.where(conf>0)[0].shape[0])

    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def matchNoBipartiteTOPK(threshold, truths, priors, variances, labels, loc_t, conf_t, idx, topk_num):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    # print('****************************************')
    least_pos_num = topk_num
    thres_low = 0.10

    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    if not best_truth_overlap.is_cuda:
        
        best_truth_overlap = best_truth_overlap.cuda()
        best_truth_idx = best_truth_idx.cuda()
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] 

    # topk anchor matching 
      
    # best_truth_idx = np.array(best_truth_idx)
    # best_truth_overlap = np.array(best_truth_overlap)
    for i in range(truths.shape[0]):
        first_idx = np.where(best_truth_idx==i)[0]
        second_idx = np.where(best_truth_overlap[first_idx]>threshold)[0]
        pick_num = second_idx.shape[0]
        # print('Before anchor matching, topk is ',pick_num)

        #if pick_num == 0:
            #print('pick_num == 0')
        if pick_num < least_pos_num:
            # the index that above thres 0.1
            #print('Before anchor matching, topk is ',pick_num)
            first_idx = np.where(best_truth_idx==i)[0]
            second_idx = np.where(best_truth_overlap[first_idx]>=thres_low)[0]
            pick_num = second_idx.shape[0]
            if pick_num >= least_pos_num:
                pick_sort = np.argsort(-best_truth_overlap[first_idx[second_idx]])
                sorted_idx = first_idx[second_idx][pick_sort]
                conf[sorted_idx[0:least_pos_num]] = 1
            else:
                conf[first_idx[second_idx]] = 1
            # pick_num = sorted_idx[0:least_pos_num].shape[0]
            #print('After anchor matching, topk is ',pick_num)
            #print('iou_value is ',best_truth_overlap[sorted_idx[0:least_pos_num]])
            #print('iou_idx is ',sorted_idx[0:least_pos_num])
        else:
            conf[first_idx[second_idx]] = 1

    # print('labels[best_truth_idx] : ',labels[best_truth_idx][0:100])
    # print('v2 --- gt_num : ',truths.shape[0])
    # print('v2 --- total conf : ',np.where(conf>0)[0].shape[0])
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def matchNoBipartiteIgnore(threshold_neg, threshold_pos, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    # print('****************************************')
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    if not best_truth_overlap.is_cuda:
        
        best_truth_overlap = best_truth_overlap.cuda()
        best_truth_idx = best_truth_idx.cuda()
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)


    # for i in range(truths.shape[0]):
    #     meet_cond_indx = np.where(best_truth_overlap[np.where(best_truth_idx==i)[0]]>=threshold)[0]
    #     pick_num = meet_cond_indx.shape[0]
    #     print('Before anchor matching, topk is ',pick_num)

    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap <= threshold_neg] = 0  # label as background
    conf[(best_truth_overlap > threshold_neg) & (best_truth_overlap < threshold_pos)] = -1
    # (max_ious>0.4) & (max_ious<0.5)
    # above_thres = np.where(best_truth_overlap >= threshold)
    # print('above_thres : ',above_thres)
    # print('size of above_thres : ',above_thres[0].shape[0])

    # print('labels[best_truth_idx] : ',labels[best_truth_idx][0:100])
    # print('v1 --- gt_num : ',truths.shape[0])
    # print('v1 --- total conf : ',np.where(conf>0)[0].shape[0])

    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances, or_img_shape):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    g_cxcy /= priors[:, 2:]
    g_wh = (matched[:, 2:] - matched[:, :2] + 1) / priors[:, 2:]
    g_wh = torch.log(g_wh)
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances, img_w, img_h):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:])), 1)

    boxes[:, 0] -= (boxes[:,2] - 1 ) / 2
    boxes[:, 1] -= (boxes[:,3] - 1 ) / 2
    boxes[:, 2] += boxes[:,0] - 1  
    boxes[:, 3] += boxes[:,1] - 1 

    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.mean()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    bool_test = False
    if bool_test:
        print('shape of scores: ',scores.shape)
        print('shape of idx: ',idx.shape)
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
