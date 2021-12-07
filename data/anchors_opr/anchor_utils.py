import numpy as np
import torch

def normalize_anchor(anchors):
    '''
    from  [c_x, cy, w, h] to [x0, x1, y0, y1] 
    '''
    return np.concatenate((anchors[:, :2] - (anchors[:, 2:] - 1) / 2,
                            anchors[:, :2] + (anchors[:, 2:] - 1) / 2), axis=1) 

def transform_anchor_opr(anchors):
    '''
    from [x0, x1, y0, y1] to [c_x, cy, w, h]
    x1 = x0 + w - 1
    c_x = (x0 + x1) / 2 = (2x0 + w - 1) / 2 = x0 + (w - 1) / 2
    '''
    return torch.cat(((anchors[:, :2] + anchors[:, 2:]) / 2 , anchors[:, 2:] - anchors[:, :2] + 1), 1)

def encode_opr(anchors, anchor_matched_gt):
    """
    anchors: Tensor (cx, cy, w, h)
    anchor_matched_gt Tensor (x0, y0, x1, y1)
    ret: Tensor (delta(cx) / anchors_w, delta(cy) / anchors_h, log(gt_w/anchor_w), log(gt_h/gt_w))
    """
    cxcy = (anchor_matched_gt[:, :2] + anchor_matched_gt[:, 2:]) / 2 - anchors[:, :2]
    cxcy /= anchors[:, 2:]

    wh = (anchor_matched_gt[:, 2:] - anchor_matched_gt[:, :2] + 1) / anchors[:, 2:]
    wh = torch.log(wh)

    return torch.cat([cxcy, wh], 1)

def transform_anchor(anchors):
    '''
    from [x0, x1, y0, y1] to [c_x, cy, w, h]
    x1 = x0 + w - 1
    c_x = (x0 + x1) / 2 = (2x0 + w - 1) / 2 = x0 + (w - 1) / 2
    '''
    return np.concatenate(((anchors[:, :2] + anchors[:, 2:]) / 2 , anchors[:, 2:] - anchors[:, :2] + 1), axis=1)
            
def get_k_center_anchor(gt_bbox, transformed_anchors, top_k):
    '''
    gt_bbox: [x0, x1, y0, y1]
    transformed_anchors: [c_x, cy, w, h]
    '''
    center_gt_bbox = transform_anchor_opr(gt_bbox.unsqueeze(0))
    l2_distance = ((transformed_anchors[:,:2] - center_gt_bbox[:,:2]) ** 2).sum(1)
    ret_idx = torch.sort(l2_distance)[1][:top_k]
    return l2_distance, ret_idx

def ig_outlier_negative_anchor(gt_bbox, transformed_anchors, ratio=1):
    '''
    gt_bbox: [x0, x1, y0, y1]
    transformed_anchors: [c_x, cy, w, h]
    '''
    scale = torch.sqrt(torch.abs((gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1))) * ratio
    center_gt_bbox = transform_anchor_opr(gt_bbox.unsqueeze(0))
    l2_distance = torch.sqrt(((transformed_anchors[:,:2] - center_gt_bbox[:,:2]) ** 2).sum(1))

    ret_idx = torch.where(l2_distance < scale)[0]
    return ret_idx


def compute_center_distance(gt_bboxes, transformed_anchors):
    '''
    gt_bboxes: n * [x0, y0, x1, y1]
    transformed_anchors: [c_x, cy, w, h]
    '''
    gt_bboxes = gt_bboxes[:, :4]
    center_gt_bboxes = transform_anchor_opr(gt_bboxes)
    l2_distance = torch.sqrt(((transformed_anchors[:,:2].unsqueeze(1) - center_gt_bboxes[:,:2].unsqueeze(0)) ** 2).sum(2))
    #ret_idx = torch.where(l2_distance.min(1)[0] > distance)[0]
    return l2_distance

def ig_outlier_negative_anchor_2(gt_bboxes, transformed_anchors, distance):
    '''
    gt_bboxes: n * [x0, x1, y0, y1]
    transformed_anchors: [c_x, cy, w, h]
    '''
    gt_bboxes = gt_bboxes[:, :4]
    center_gt_bboxes = transform_anchor_opr(gt_bboxes)
    l2_distance = torch.sqrt(((transformed_anchors[:,:2].unsqueeze(1) - center_gt_bboxes[:,:2].unsqueeze(0)) ** 2).sum(2))
    ret_idx = torch.where(l2_distance.min(1)[0] > distance)[0]
    return ret_idx

def encode(anchors, anchor_matched_gt):
    """
    anchors: np.array (cx, cy, w, h)
    anchor_matched_gt np.array (x0, y0, x1, y1)
    ret: np.array (delta(cx) / anchors_w, delta(cy) / anchors_h, log(gt_w/anchor_w), log(gt_h/gt_w))
    """
    cxcy = (anchor_matched_gt[:, :2] + anchor_matched_gt[:, 2:]) / 2 - anchors[:, :2]
    cxcy /= anchors[:, 2:]

    wh = (anchor_matched_gt[:, 2:] - anchor_matched_gt[:, :2] + 1) / anchors[:, 2:]
    wh = np.log(wh)
    
    return np.concatenate([cxcy, wh], axis=1)
  
def decode(loc, anchors):
    """
    loc: torch.Tensor
    anchors: 2-d, torch.Tensor (cx, cy, w, h)
    boxes: 2-d, torch.Tensor (x0, y0, x1, y1)
    """

    boxes = torch.cat((
        anchors[:, :2] + loc[:, :2] * anchors[:, 2:],
        anchors[:, 2:] * torch.exp(loc[:, 2:])), 1)

    boxes[:, 0] -= (boxes[:,2] - 1 ) / 2
    boxes[:, 1] -= (boxes[:,3] - 1 ) / 2
    boxes[:, 2] += boxes[:,0] - 1  
    boxes[:, 3] += boxes[:,1] - 1 

    return boxes

