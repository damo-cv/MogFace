# ******************************************************
# Author        : liuyang
# Last modified : 2020-01-14 14:16
# Email         : gxly1314@gmail.com
# Filename      : bbox_utils.py
# Description   : 
# ******************************************************
import numpy as np
import torch
def intersect(box_a, box_b, normalized=True):
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = np.minimum(np.broadcast_to(box_a[:, 2:][:, np.newaxis, :], (A,B,2)), \
                        np.broadcast_to(box_b[:, 2:][np.newaxis, :, :], (A,B,2)))
    min_xy = np.maximum(np.broadcast_to(box_a[:, :2][:, np.newaxis, :], (A,B,2)), \
                        np.broadcast_to(box_b[:, :2][np.newaxis, :, :], (A,B,2)))

    if normalized:
        max_xy[:, :, 0] = max_xy[:,:,0] - min_xy[:,:,0] + 1
        max_xy[:, :, 1] = max_xy[:,:,1] - min_xy[:,:,1] + 1
    else:
        max_xy[:, :, 0] = max_xy[:,:,0] - min_xy[:,:,0] 
        max_xy[:, :, 1] = max_xy[:,:,1] - min_xy[:,:,1]

    max_xy = np.clip(max_xy, 0, float('inf'))
    res = max_xy[:, :, 0] * max_xy[:, :, 1]

    return res

def bbox_overlap(box_a, box_b, normalized=True):
    '''
    Args:
    box_a : np.ndarray [n, 4] or [1,4] or [4,] x0, y0, x1, y1 
    box_b : np.ndarray [k, 4] or [1,4] or [4,] x0, y0, x1, y1
    normalized: computer_area = (x1 - x0 + 1) * (y1 - y0 + 1)
    Ret:
    iou: 2-d or 1-d 
    '''
    squeeze_final_tensor = False

    if len(box_a.shape) == 1:
        box_a = box_a[:, np.newaxis, :]
        squeeze_final_tensor = True
    if len(box_b.shape) == 1:
        box_b = box_b[np.newaxis, :]
        squeeze_final_tensor = True

    assert len(box_a.shape) == 2 and len(box_b.shape) == 2, 'only support compute 2-d tensor overlap'
    inter = intersect(box_a, box_b, normalized)

    if normalized:
        area_a = np.broadcast_to(((box_a[:, 2] - box_a[:, 0] + 1) * \
                                  (box_a[:, 3] - box_a[:, 1] + 1))[:, np.newaxis], \
                                  (inter.shape[0], inter.shape[1]))

        area_b = np.broadcast_to(((box_b[:, 2] - box_b[:, 0] + 1) * \
                                   (box_b[:, 3] - box_b[:, 1] + 1))[np.newaxis,:], \
                                   (inter.shape[0], inter.shape[1]))
    else:
        area_a = np.broadcast_to(((box_a[:, 2] - box_a[:, 0]) * \
                                  (box_a[:, 3] - box_a[:, 1]))[:, np.newaxis], \
                                  (inter.shape[0], inter.shape[1]))

        area_b = np.broadcast_to(((box_b[:, 2] - box_b[:, 0]) * \
                                  (box_b[:, 3] - box_b[:, 1]))[np.newaxis,:], \
                                  (inter.shape[0], inter.shape[1]))
         

    union = area_a + area_b - inter
    if squeeze_final_tensor:
        return (inter / union).squeeze()
    else:
        return inter / union

def intersect_opr(box_a, box_b, normalized=True):
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A,B,2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A,B,2))
    if normalized:
        max_xy[:, :, 0] = max_xy[:,:,0] - min_xy[:,:,0] + 1
        max_xy[:, :, 1] = max_xy[:,:,1] - min_xy[:,:,1] + 1
    else:
        max_xy[:, :, 0] = max_xy[:,:,0] - min_xy[:,:,0] 
        max_xy[:, :, 1] = max_xy[:,:,1] - min_xy[:,:,1]

    max_xy = torch.clamp(max_xy, 0, float('inf'))
    res = max_xy[:, :, 0] * max_xy[:, :, 1]

    return res

def bbox_overlap_opr(box_a, box_b, normalized=True):
    '''
    Args:
    box_a : torch.FloatTensor [n, 4] or [1,4] or [4,] x0, y0, x1, y1 
    box_b : torch.FloatTensor [k, 4] or [1,4] or [4,] x0, y0, x1, y1
    normalized: computer_area = (x1 - x0 + 1) * (y1 - y0 + 1)
    Ret:
    iou: 2-d or 1-d 
    '''
    squeeze_final_tensor = False

    if len(box_a.shape) == 1:
        box_a = box_a[:, np.newaxis, :]
        squeeze_final_tensor = True
    if len(box_b.shape) == 1:
        box_b = box_b[np.newaxis, :]
        squeeze_final_tensor = True

    assert len(box_a.shape) == 2 and len(box_b.shape) == 2, 'only support compute 2-d tensor overlap'
    inter = intersect_opr(box_a, box_b, normalized)

    if normalized:
        area_a = ((box_a[:, 2] - box_a[:, 0] + 1) * (box_a[:, 3] - box_a[:, 1] + 1)) \
                  .unsqueeze(1).expand(inter.shape[0], inter.shape[1])

        area_b = ((box_b[:, 2] - box_b[:, 0] + 1) * (box_b[:, 3] - box_b[:, 1] + 1)) \
                  .unsqueeze(0).expand(inter.shape[0], inter.shape[1])
    else:
        area_a = np.broadcast_to(((box_a[:, 2] - box_a[:, 0]) * \
                                  (box_a[:, 3] - box_a[:, 1]))[:, np.newaxis], \
                                  (inter.shape[0], inter.shape[1]))

        area_b = np.broadcast_to(((box_b[:, 2] - box_b[:, 0]) * \
                                  (box_b[:, 3] - box_b[:, 1]))[np.newaxis,:], \
                                  (inter.shape[0], inter.shape[1]))
         

    union = area_a + area_b - inter
    if squeeze_final_tensor:
        return (inter / union).squeeze()
    else:
        return inter / union


if __name__ == '__main__':
    anchors = np.arange(20).reshape(5,4)
    gt_boxes = np.arange(12).reshape(3,4)
    torch_anchors = torch.Tensor(anchors).float()
    torch_gt_bboxes = torch.Tensor(gt_boxes).float()
    torch_overlap = bbox_overlap_opr(torch_anchors, torch_gt_bboxes)
    overlap = bbox_overlap(anchors, gt_boxes)

    gt_boxes_single  = np.arange(4).reshape(-1)
    torch_gt_boxes_single  = torch.Tensor(np.arange(4).reshape(-1)).float()
    #overlap_1 = bbox_overlap(anchors, gt_boxes_single[np.newaxis, :])
    overlap_2 = bbox_overlap(anchors, gt_boxes_single)
    torch_overlap_2 = bbox_overlap_opr(torch_anchors, torch_gt_boxes_single)
    #overlap_2 = bbox_overlap(anchors, gt_boxes_single, normalized=False)
    #overlap_3 = jaccard_numpy(anchors, gt_boxes_single)
    import pdb;pdb.set_trace()
