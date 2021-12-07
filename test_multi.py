# ******************************************************
# Author       : liuyang
# Last modified:	2020-01-15 15:54
# Email        : gxly1314@gmail.com
# Filename     :	new_test.py
# Description  : 
# ******************************************************
from __future__ import absolute_import
import sys
import argparse
import numpy as np
import torch
import scipy.io as sio
import datetime
import os
import cv2
import torch.backends.cudnn as cudnn
from core.workspace import register, create, global_config, load_config
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
from utils.nms.nms_wrapper import nms
from data import anchor_utils
from tqdm import tqdm
from evaluation.evaluate_ap50 import evaluation_ap50

parser = argparse.ArgumentParser(description='Test Details')
parser.add_argument('--num_iter', '-n', default=140, type=int, help='number of iteration for test.')
parser.add_argument('--nms_th', default=0.3, type=float, help='nms threshold.')
parser.add_argument('--pre_nms_top_k', default=5000, type=int, help='number of max score image.')
parser.add_argument('--score_th', default=0.01, type=float, help='score threshold.')
parser.add_argument('--max_bbox_per_img', default=750, type=int, help='max number of det bbox.')
parser.add_argument('--scale_weight', default=15, type=float, help='to differentiate the gap between large and small scale..')
parser.add_argument('--max_img_shrink', default=2.6, type=float, help='constrain the max shrink of img.')
parser.add_argument('--vote_th', default=0.6, type=float, help='bbox vote threshold')
parser.add_argument('--config', '-c', default='./config.yml', type=str, help='config yml.')
parser.add_argument('--sub_project_name', default=None, type=str, help='sub_project_name.')
parser.add_argument('--test_min_scale', default=0, type=int, help='the min scale of det bbox')
parser.add_argument('--flip_ratio', default=None, type=float)
parser.add_argument('--test_hard', default=0, type=int)



def detect_face(image, shrink):
    # starttime = datetime.datetime.now()
    x = image
    if shrink != 1:
        x = cv2.resize(image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)


    print('shrink:{}'.format(shrink))

    width = x.shape[1]
    height = x.shape[0]
    print('width: {}, height: {}'.format(width, height))

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = Variable(x.cuda(), volatile=True)

    out = net(x)
    
    anchors = anchor_utils.transform_anchor((val_set.generate_anchors_fn(height, width)))
    anchors = torch.FloatTensor(anchors).cuda()
    decode_bbox =  anchor_utils.decode(out[1].squeeze(0), anchors)
    boxes = decode_bbox
    scores = out[0].squeeze(0)

    top_k = args.pre_nms_top_k
    v, idx = scores[:, 0].sort(0)
    idx = idx[-top_k:]
    boxes = boxes[idx]
    scores = scores[idx]

    # [11620, 4]
    boxes = boxes.cpu().numpy()
    w = boxes[ :, 2] - boxes[:,0] + 1
    h = boxes[ :, 3] - boxes[:,1] + 1
    boxes[:,0] /= shrink
    boxes[:,1] /= shrink
    boxes[:,2] = boxes[:,0] + w / shrink - 1
    boxes[:,3] = boxes[:,1] + h / shrink - 1
    #boxes = boxes / shrink
    # [11620, 2]
    if int(args.test_min_scale) != 0 :
        boxes_area = (boxes[:, 3] - boxes[:, 1] + 1) * (boxes[:, 2] - boxes[:, 0] + 1) /  (shrink * shrink)
        boxes = boxes[boxes_area >  int(args.test_min_scale)**2]
        scores = scores[boxes_area > int(args.test_min_scale)**2]

    scores = scores.cpu().numpy()

    inds = np.where(scores[:, 0] > args.score_th)[0]
    if len(inds) == 0:
        det = np.empty([0, 5], dtype=np.float32)
        return det
    c_bboxes = boxes[inds]
    # [5,]
    c_scores = scores[inds, 0]
    # [5, 5]
    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
    
    #starttime = datetime.datetime.now()
    keep = nms(c_dets, args.nms_th)
    #endtime = datetime.datetime.now()
    #print('nms forward time = ',(endtime - starttime).seconds+(endtime - starttime).microseconds/1000000.0,' s')
    c_dets = c_dets[keep, :]

    max_bbox_per_img = args.max_bbox_per_img
    if max_bbox_per_img > 0:
        image_scores = c_dets[:, -1]
        if len(image_scores) > max_bbox_per_img:
            image_thresh = np.sort(image_scores)[-max_bbox_per_img]
            keep = np.where(c_dets[:, -1] >= image_thresh)[0]
            c_dets = c_dets[keep, :]
    return c_dets

def multi_scale_test(image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(image, st)
    if max_im_shrink > 0.75:
        det_s = np.row_stack((det_s,detect_face(image,0.75)))
    #index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    if args.scale_weight == -1:
        index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    else:
        index = np.where(((det_s[:, 2] - det_s[:, 0]) * (det_s[:, 3] - det_s[:, 1])) > 2000)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(image, bt)
    if args.scale_weight == -1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
    else:
        index = np.where(((det_b[:, 2] - det_b[:, 0]) * (det_b[:, 3] - det_b[:, 1])) < args.scale_weight  * 600 )[0]
    det_b = det_b[index,:]

    # enlarge small iamge x times for small face
    if max_im_shrink > 1.5:
        det_tmp = detect_face(image,1.5)
        if args.scale_weight == -1:
            index = np.where(np.minimum(det_tmp[:, 2] - det_tmp[:, 0] + 1, det_tmp[:, 3] - det_tmp[:, 1] + 1) < 100)[0]
        else:
            index = np.where(((det_tmp[:, 2] - det_tmp[:, 0]) * (det_tmp[:, 3] - det_tmp[:, 1])) < args.scale_weight * 800 )[0]
        det_tmp = det_tmp[index, :]
        det_b = np.row_stack((det_b, det_tmp))

    if max_im_shrink > 2:
        det_tmp = detect_face(image, max_im_shrink)
        if args.scale_weight == -1:
            index = np.where(np.minimum(det_tmp[:, 2] - det_tmp[:, 0] + 1, det_tmp[:, 3] - det_tmp[:, 1] + 1) < 100)[0]
        else:
            index = np.where(((det_tmp[:, 2] - det_tmp[:, 0]) * (det_tmp[:, 3] - det_tmp[:, 1])) < args.scale_weight * 500)[0]
        det_tmp = det_tmp[index, :]
        det_b = np.row_stack((det_b, det_tmp))


    return det_s, det_b

def multi_scale_test_pyramid(image, max_shrink):
    # shrink detecting and shrink only detect big face
    det_b = detect_face(image, 0.25)
    if args.scale_weight == -1:
        index = np.where(
                np.maximum(det_b[:, 2] - det_b[:, 0] + 1,
                det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
    else:
        index = np.where(((det_b[:, 2] - det_b[:, 0]) * (det_b[:, 3] - det_b[:, 1])) >  2000 )[0]
    det_b = det_b[index, :]

    st = [1.25, 1.75, 2.25]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            det_temp = detect_face(image, st[i])
            # enlarge only detect small face
            if i == 0:
                if args.scale_weight == -1:
                    index = np.where(
                            np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                            det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                else:
                    index = np.where(((det_temp[:, 2] - det_temp[:, 0]) * (det_temp[:, 3] - det_temp[:, 1])) < args.scale_weight  * 2000)[0]
                det_temp = det_temp[index,:]
            if i == 1:
                if args.scale_weight == -1:
                    index = np.where(
                            np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                            det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                else:
                    index = np.where(((det_temp[:, 2] - det_temp[:, 0]) * (det_temp[:, 3] - det_temp[:, 1])) < args.scale_weight  * 1000)[0]
                det_temp = det_temp[index,:]
            if i == 2:
                if args.scale_weight == -1:
                    index = np.where(
                            np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                            det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                else:
                    index = np.where(((det_temp[:, 2] - det_temp[:, 0]) * (det_temp[:, 3] - det_temp[:, 1])) < args.scale_weight  *  600)[0]
                det_temp = det_temp[index,:]
            det_b = np.row_stack((det_b, det_temp))


    return det_b

def flip_test(image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2] - 1
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0] - 1
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t

def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    det[:,:4] = np.round(det[:,:4]) 
    while det.shape[0] > 0:
        # IOU
        box_w = np.maximum(det[:, 2] - det[:, 0], 0)
        box_h = np.maximum(det[:, 3] - det[:, 1], 0)
        area = box_w * box_h
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = area[0] + area[:] - inter
        union[union <=0] = 1
        o = inter / union
        o[0] = 1

        # get needed merge det and delete these det
        merge_index = np.where(o >= args.vote_th)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    if dets.shape[0] > 750:
       dets = dets[0:750, :]
    return dets


def write_to_txt(f, det, height, width, img_name, img_dir_name):
    f.write('{:s}\n'.format(img_dir_name + '/' + img_name + '.jpg'))
    f.write('{:d}\n'.format(det.shape[0]))
    for i in range(det.shape[0]):
        if det[i][0] < 0.0:
            xmin = 0.0
        else:
            xmin = det[i][0]

        if det[i][1] < 0.0:
            ymin = 0.0
        else:
            ymin = det[i][1]

        if det[i][2] > width - 1:
            xmax = width - 1
        else:
            xmax = det[i][2]

        if det[i][3] > height - 1:
            ymax = height - 1
        else:
            ymax = det[i][3]

        score = det[i][4]
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                format(round(xmin), round(ymin), round(xmax - xmin + 1), round(ymax - ymin + 1), score))

def gen_soft_link_dir(dir_name_list):
    for dir_name in dir_name_list:
        cur_dir_name = dir_name.split('/')[-1]
        if os.path.exists(cur_dir_name):
            os.system('rm -rf ./{}'.format(cur_dir_name))
        if not os.path.exists(dir_name):
            raise ValueError('Cannot create soft link, {} does not exist'.format(dir_name))
        os.system('ln -s {} ./{}'.format(dir_name, cur_dir_name))

def gen_dir(dir_name_list):
    for dir_name in dir_name_list:
        if not os.path.exists(dir_name):
            os.system('mkdir -p {}'.format(dir_name))

if __name__ == '__main__':
    args = parser.parse_args()
    if args.test_hard:
        args.max_img_shrink = 2.3
        args.vote_th = 0.5
        args.nms_th = 0.4
        args.scale_weight = 10
        args.flip_ratio = 1.4

    # generate det_info and det_result
    cfg = load_config(args.config)
    cfg['phase'] = 'test'

    config_name = args.config.split('/')[-1].split('.')[-2]
    snapshots_dir = os.path.join('./snapshots', config_name)

    det_info_dir = os.path.join('./det_info', config_name)

    det_result_dir = os.path.join('./det_result', config_name) 

    save_info_dir_name  = 'ss_' + str(args.num_iter) + '_nmsth_' + str(args.nms_th) + \
               '_scoreth_' + str(args.score_th) 

    abs_save_dir = os.path.join(det_info_dir, save_info_dir_name)
    det_result_txt = os.path.join(det_result_dir, 'result.txt')

    gen_dir_list = [abs_save_dir, det_result_dir]
    gen_dir(gen_dir_list)

    # create net and val_set
    net = create(cfg.architecture)
    model_name = os.path.join(snapshots_dir, 'model_{}000.pth'.format(args.num_iter))
    print ('Load model from {}'.format(model_name))
    net.load_state_dict(torch.load(model_name))
    net.cuda()
    net.eval()
    print ('Finish load model.')

    val_set= create(cfg.validation_set)
    val_set_iter = iter(val_set)

    # generate predict bbox
    for (img, img_name, img_dir_name) in tqdm(val_set_iter):
        event_dir = os.path.join(abs_save_dir, img_dir_name)
        if not os.path.exists(event_dir):
            os.system('mkdir -p {}'.format(event_dir))
        with torch.no_grad():
            max_im_shrink = (0x7fffffff / 200.0 / (img.shape[0] * img.shape[1])) ** 0.5 # the max size of input image for caffe
            max_im_shrink = args.max_img_shrink  if max_im_shrink > 2.2 else max_im_shrink
            shrink = max_im_shrink if max_im_shrink < 1 else 1
            det0 = detect_face(img, shrink)  # origin test
            det1 = flip_test(img, shrink)    # flip test
            [det2, det3] = multi_scale_test(img, max_im_shrink)
            det4 = multi_scale_test_pyramid(img, max_im_shrink)
            if args.flip_ratio is not None:
                det5 = flip_test(img, args.flip_ratio)

        if args.flip_ratio is not None:
            det = np.row_stack((det0, det1, det2, det3, det4, det5))
        else:
            det = np.row_stack((det0, det1, det2, det3, det4))
        dets = bbox_vote(det)

        save_img_name = os.path.join(event_dir, img_name + '.txt')
        with open(save_img_name, 'w') as f:
            write_to_txt(f, dets, img.shape[0], img.shape[1], img_name, img_dir_name)

    pred = abs_save_dir
    gt_path = './dataset/WIDERFACE/ground_truth'
    test_iter = args.num_iter
    det_result_txt = det_result_txt
    easy_ap, medium_ap, hard_ap = evaluation_ap50(pred, gt_path, test_iter, det_result_txt)
    print('Test Iter: {}, Easy: {}, Medium: {}, Hard: {}'.format(test_iter, easy_ap, medium_ap, hard_ap))
