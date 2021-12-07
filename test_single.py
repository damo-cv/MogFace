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
#from modelling.ops.nms.nms_wrapper import nms
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
parser.add_argument('--config', '-c', default='./config.yml', type=str, help='config yml.')
parser.add_argument('--sub_project_name', default=None, type=str, help='sub_project_name.')
parser.add_argument('--backbone_cfg_file', '-bcf', default=None, type=str, help='backbone config file')
parser.add_argument('--test_idx',  default=None, type=int)

def detect_face_with_net(net, image, shrink, val_set, gpu=None):
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
    if gpu == None:
        x = Variable(x.cuda(), volatile=True)
    else:
        x = Variable(x.cuda(gpu), volatile=True)

    out = net(x)
    
    anchors = anchor_utils.transform_anchor((val_set.generate_anchors_fn(height, width)))
    if gpu == None:
        anchors = torch.FloatTensor(anchors).cuda()
    else:
        anchors = torch.FloatTensor(anchors).cuda(gpu)
    decode_bbox =  anchor_utils.decode(out[1].squeeze(0), anchors)
    boxes = decode_bbox
    scores = out[0].squeeze(0)

    top_k = 5000
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
    scores = scores.cpu().numpy()

    inds = np.where(scores[:, 0] > 0.01)[0]
    if len(inds) == 0:
        det = np.empty([0, 5], dtype=np.float32)
        return det
    c_bboxes = boxes[inds]
    # [5,]
    c_scores = scores[inds, 0]
    # [5, 5]
    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
    
    #starttime = datetime.datetime.now()
    keep = nms(c_dets, 0.3)
    #endtime = datetime.datetime.now()
    #print('nms forward time = ',(endtime - starttime).seconds+(endtime - starttime).microseconds/1000000.0,' s')
    c_dets = c_dets[keep, :]

    max_bbox_per_img = 750
    if max_bbox_per_img > 0:
        image_scores = c_dets[:, -1]
        if len(image_scores) > max_bbox_per_img:
            image_thresh = np.sort(image_scores)[-max_bbox_per_img]
            keep = np.where(c_dets[:, -1] >= image_thresh)[0]
            c_dets = c_dets[keep, :]
    return c_dets



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

    select_idx_list = []
    tmp_height = height
    tmp_width = width

    test_idx = args.test_idx 
    if test_idx is not None:
        for i in range(2):
            tmp_height = (tmp_height + 1) // 2
            tmp_width = (tmp_width + 1) // 2

        for i in range(6):
            if i == 0:
                select_idx_list.append(tmp_height * tmp_width)
            else:
                select_idx_list.append(tmp_height * tmp_width + select_idx_list[i-1])
            tmp_height = (tmp_height + 1) // 2
            tmp_width = (tmp_width + 1) // 2

        if test_idx == 2:
            boxes = boxes[:select_idx_list[(test_idx-2)]]
            scores = scores[:select_idx_list[(test_idx-2)]]
        else:
            boxes = boxes[select_idx_list[test_idx - 3] : select_idx_list[test_idx - 2]]
            scores = scores[select_idx_list[test_idx - 3] : select_idx_list[test_idx - 2]]

    print('scores shape', scores.shape)
    print('boxes shape', boxes.shape)
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

def write_to_txt_clip_border(f, det, height, width, img_name, img_dir_name):
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
                format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))

def write_to_txt(f, det):
    f.write('{:s}\n'.format(str(event[0][0].encode('utf-8'))[2:-1] + '/' + im_name + '.jpg'))
    f.write('{:d}\n'.format(det.shape[0]))
    for i in range(det.shape[0]):
        xmin = det[i][0]
        ymin = det[i][1]
        xmax = det[i][2]
        ymax = det[i][3]
        score = det[i][4]
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))

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
    # generate det_info and det_result
    cfg = load_config(args.config)
    cfg['phase'] = 'test'
    if 'use_hcam' in cfg and cfg['use_hcam']:
        # test_th
        cfg['fp_th'] = 0.12

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
            max_im_shrink = 2.2 if max_im_shrink > 2.2 else max_im_shrink
            shrink = max_im_shrink if max_im_shrink < 1 else 1
            det0 = detect_face(img, shrink)  # origin test
        save_img_name = os.path.join(event_dir, img_name + '.txt')
        with open(save_img_name, 'w') as f:
            write_to_txt_clip_border(f, det0, img.shape[0], img.shape[1], img_name, img_dir_name)

    pred = abs_save_dir
    gt_path = './dataset/WIDERFACE/ground_truth'
    test_iter = args.num_iter
    det_result_txt = det_result_txt
    easy_ap, medium_ap, hard_ap = evaluation_ap50(pred, gt_path, test_iter, det_result_txt)
    print('Test Iter: {}, Easy: {}, Medium: {}, Hard: {}'.format(test_iter, easy_ap, medium_ap, hard_ap))
