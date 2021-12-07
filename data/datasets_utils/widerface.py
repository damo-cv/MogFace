# ******************************************************
# Author       : liuyang
# Last modified: 2020-01-13 20:28
# Email        : gxly1314@gmail.com
# Filename     : widerface.py
# Description  : 
# ******************************************************
from __future__ import absolute_import
import cv2
import os
import numpy as np 
from tools.visualize import draw_bboxes
from .map_dict import wider_map_dict
from ..transform import image_util
import copy
from ..anchors_opr import anchor_utils
from core.workspace import register
import scipy.io as sio
cv2.setNumThreads(0)

@register
class WiderFaceValSet(object):
    __shared__ = ['generate_anchors_fn', 'preprocess_fn']
    __inject__ = ['generate_anchors_fn', 'preprocess_fn']
    def __init__(self, 
                 phase='test',
                 base_data_path = './dataset',
                 img_info_mat_path='WIDERFACE/wider_face_split/wider_face_val.mat',
                 img_dir_name='WIDERFACE/WIDER_val/images',
                 generate_anchors_fn=None, 
                 preprocess_fn=None,
                 ):
        self.phase=phase
        self.dataset_path = base_data_path
        self.img_info_mat_path  = os.path.join(self.dataset_path, img_info_mat_path)
        self.img_dir_name = os.path.join(self.dataset_path, img_dir_name)
        self.generate_anchors_fn = generate_anchors_fn
        self.preprocess_fn = preprocess_fn
        self.img_info_list = self.parse_gt_mat()
        self.img_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        '''
        return img, img_name(not abs_path), dir_name(not abs_path)
        e.g. img_name: []
        '''
        if self.img_idx >= len(self):
            raise StopIteration
        item = self.img_info_list[self.img_idx]
        abs_img_path = os.path.join(self.img_dir_name, item[1], item[0]) + '.jpg'
        img = cv2.imread(abs_img_path).astype(np.float32)
        img = self.preprocess_fn(img, phase=self.phase)
        self.img_idx += 1
        return img, item[0], item[1]


    def __getitem__(self, idx):
        return self.img_info_list[idx]
        pass

    def __len__(self):
        return len(self.img_info_list)

    def parse_gt_mat(self):
        '''
        ret: [[img_name(not abs_path), dir_name(not abs_path)], ...]
        '''
        wider_face = sio.loadmat(self.img_info_mat_path)
        event_list = wider_face['event_list']
        file_list = wider_face['file_list']
        ret_list = []
        del wider_face
        for index, event in enumerate(event_list):
            filelist = file_list[index][0]
            for num, file in enumerate(filelist):
                im_name = str(file[0][0].encode('utf-8'))[2:-1] 
                dir_name = str(event[0][0].encode('utf-8'))[2:-1]
                ret_list.append([im_name, dir_name])

        return ret_list

                
@register
class WiderFaceTrainSet(object):
    '''
    TODO: if ignore training no bbox img, rewrite load_file_list
    '''
    __shared__ = ['generate_anchors_fn', 'preprocess_fn', 'out_bbox_anchor']
    __inject__ = ['generate_anchors_fn', 'anchor_target_fn', 'preprocess_fn', \
                  'data_aug_settings']
    def __init__(self,
                phase='training',
                debug_img_dir='./debug_img_dir',
                base_data_path = './dataset',
                gt_file='WIDERFACE/wider_face_split/wider_face_train_bbx_gt.txt',
                img_dir_name='WIDERFACE/WIDER_train/images',
                generate_anchors_fn=None,
                anchor_target_fn=None,
                preprocess_fn=None,
                data_aug_settings=None,
                out_bbox_anchor=False,
                gt_list=None,
                out_dsfd_label=False):
        self.phase=phase
        self.gt_idx = 0
        self.map_dict = wider_map_dict
        self.debug_img_dir = debug_img_dir
        self.generate_anchors_fn = generate_anchors_fn
        self.anchor_target_fn = anchor_target_fn
        self.preprocess_fn = preprocess_fn
        self.data_aug_settings = data_aug_settings
        self.base_data_path = base_data_path
        self.gt_file = os.path.join(base_data_path, gt_file)
        self.img_dir_name = os.path.join(base_data_path, img_dir_name)
        self.out_bbox_anchor = out_bbox_anchor

        self.out_dsfd_label = out_dsfd_label

        if self.data_aug_settings.use_mst:
            train_scale = np.array([640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024,1056,1088,1120,1152,1184], dtype=np.int)
            single_scale = np.random.choice(train_scale)
            if self.data_aug_settings.resize_height:
                self.data_aug_settings.resize_height = single_scale
            if self.data_aug_settings.resize_width:
                self.data_aug_settings.resize_width = single_scale
            if self.data_aug_settings.sample_bbox_width:
                self.data_aug_settings.sample_bbox_width = single_scale
            if self.data_aug_settings.sample_bbox_height:
                self.data_aug_settings.sample_bbox_height = single_scale

        self.anchors = self.generate_anchors_fn(self.data_aug_settings.resize_height, \
                                                self.data_aug_settings.resize_width)
        if self.out_dsfd_label:
            generate_dsfd_anchors_fn = copy.deepcopy(self.generate_anchors_fn)
            generate_dsfd_anchors_fn.anchor_size_list = [8, 16, 32, 64, 128, 256]
            self.dsfd_anchors = generate_dsfd_anchors_fn(self.data_aug_settings.resize_height, \
                                        self.data_aug_settings.resize_width)


        if gt_list is not None:
            self.gt_list = gt_list
        else:
            self.gt_list = self.load_file_list(self.gt_file)

    def __len__(self):
        return len(self.gt_list)


    def __getitem__(self, gt_idx):
        '''
        bbox_labels(list) : xmin | ymin | xmax | ymax | label
        '''
        bbox_labels = self.parse_gt_list(self.gt_list, gt_idx)
        img_name = self.gt_list[gt_idx][0]
        img_path = os.path.join(self.img_dir_name, img_name)
        img = cv2.imread(img_path).astype('float32')

        if self.data_aug_settings.use_mst:
            img, bbox_labels = self.preprocess_fn(img, bbox_labels, self.phase, self.data_aug_settings)
        else:
            img, bbox_labels = self.preprocess_fn(img, bbox_labels, self.phase)

        if bbox_labels.shape[0] == 0:
            if self.out_dsfd_label:
                return img, np.zeros((self.anchors.shape[0], 5)).astype('float32'), \
                             np.zeros((self.anchors.shape[0], 5)).astype('float32')
            elif self.out_bbox_anchor:
                return img, np.zeros((self.anchors.shape[0], 5)).astype('float32'), self.anchors,\
                       np.zeros((self.anchors.shape[0], 5)).astype('float32')
            else:
                return img, np.zeros((self.anchors.shape[0], 5)).astype('float32')
        encode_bbox_labels = self.anchor_target_fn(self.anchors, bbox_labels.astype('float32'))
        #self.gt_idx += 1

        if self.out_dsfd_label:
            encode_dsfd_bbox_labels = self.anchor_target_fn(self.dsfd_anchors, bbox_labels.astype('float32'))
            return img, encode_bbox_labels, encode_dsfd_bbox_labels
        elif self.out_bbox_anchor:
            return img, encode_bbox_labels, self.anchors, bbox_labels
        else:
            return img, encode_bbox_labels

    def validate(self, num_validated_img=100, shuffle=True):
        import torch
        num_validated_img = min(len(self.gt_list), num_validated_img)
        print('validate img, bbox_labels by randomly samlping {} img and saved in {}.'\
              .format(num_validated_img, self.debug_img_dir))
        gt_list = self.gt_list.copy()
        if shuffle:
            np.random.shuffle(gt_list)
        if not os.path.exists(self.debug_img_dir):
            os.system('mkdir {}'.format(self.debug_img_dir))
        else:
            os.system('rm -rf {}'.format(os.path.join(self.debug_img_dir, '*')))

        for idx,item in enumerate(gt_list):
            if idx > num_validated_img - 1:
                break
            img_name = item[0]
            img_path = os.path.join(self.img_dir_name, img_name)
            img = cv2.imread(img_path).astype('float32')
            # validate ori_img
            bbox_labels = self.parse_gt_list(gt_list, idx)
            labels = np.array(bbox_labels)[:,4].tolist()
            label_list = []
            for label in labels:
                label_list.append(self.map_dict[str(int(label))])

            draw_bboxes(img, np.array(bbox_labels)[:, :4], label_list, output_dir=self.debug_img_dir, \
                        save_img_name='or_img_{}_{}'.format(np.random.uniform(0,1), img_name.split('/')[-1]))

            # validate data aug mentation.
            img, bbox_labels = self.preprocess_fn(img, bbox_labels, self.phase)
            labels = np.array(bbox_labels)[:,4].tolist()
            label_list = []
            for label in labels:
                label_list.append(self.map_dict[str(int(label))])

            draw_bboxes(img, np.array(bbox_labels)[:, :4], label_list, output_dir=self.debug_img_dir, \
                        save_img_name='data_aug_img_{}_{}'.format(np.random.uniform(0,1), img_name.split('/')[-1]))

            # validate decode fn.
            encode_bbox_labels = self.anchor_target_fn(self.anchors, bbox_labels.astype('float32'))
            decode_bboxes = anchor_utils.decode(torch.Tensor(encode_bbox_labels[:,:4]), \
                                                torch.Tensor(anchor_utils.transform_anchor(self.anchors)))
            decode_bboxes = decode_bboxes.numpy()
            decode_bboxes_pos = decode_bboxes[np.where(encode_bbox_labels[:,-1] == 1)]
            draw_bboxes(img, np.array(decode_bboxes_pos), output_dir=self.debug_img_dir, \
              save_img_name='decode_img_{}_{}'.format(np.random.uniform(0,1), img_name.split('/')[-1]))

    @staticmethod
    def load_file_list(gt_file):
        """
        return [[img_name, num_face, [x0, y0, w, h], [...], [...]...], ...]
        TODO: the meaning of other bool value in input_txt, e.g. is_blur

        """
        with open(gt_file, 'r') as f_dir:
            lines_input_txt = f_dir.readlines()

        file_dict = {}
        num_class = 0
        for i in range(len(lines_input_txt)):
            line_txt = lines_input_txt[i].strip('\n\t\r')
            if '--' in line_txt:
                if i != 0:
                    num_class += 1
                file_dict[num_class] = []
                file_dict[num_class].append(line_txt)
            if '--' not in line_txt:
                if len(line_txt) > 6:
                    split_str = line_txt.split(' ')
                    x1_min = float(split_str[0])
                    y1_min = float(split_str[1])
                    w = float(split_str[2])
                    h = float(split_str[3])
                    line_txt = str(x1_min) + ' ' + str(y1_min) + ' ' + str(
                        w) + ' ' + str(h)
                    file_dict[num_class].append(line_txt)
                else:
                    file_dict[num_class].append(line_txt)

        return list(file_dict.values())

    @staticmethod
    def parse_gt_list(gt_list, gt_idx):
        """
        return : xmin | ymin | xmax | ymax | label
        """
        item = gt_list[gt_idx]
        bbox_labels = []
        for index_box in range(len(item)):
            if index_box >= 2:
                bbox_sample = []
                temp_info_box = item[index_box].split(' ')
                xmin = int(float(temp_info_box[0]))
                ymin = int(float(temp_info_box[1]))
                w = int(float(temp_info_box[2]))
                h = int(float(temp_info_box[3]))
                xmax = xmin + w - 1 
                ymax = ymin + h - 1
                # Filter out wrong labels
                if w == 0 or h == 0:
                    continue
                elif w < 0:
                    tmp = xmin
                    xmin = xmax
                    xmax = xmin
                elif h < 0:
                    tmp = ymin
                    ymin = ymax
                    ymax = tmp

                bbox_sample.append(float(xmin))
                bbox_sample.append(float(ymin))
                bbox_sample.append(float(xmax))
                bbox_sample.append(float(ymax))
                bbox_sample.append(1)
                bbox_labels.append(bbox_sample)

        return np.array(bbox_labels)

