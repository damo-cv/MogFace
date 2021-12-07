# ******************************************************
# Author        : liuyang
# Last modified : 2020-01-13 20:46
# Email         : gxly1314@gmail.com
# Filename      : generate_anchors.py
# Description   : 
# ******************************************************
import numpy as np 
import math
# TODO add faster-rcnn generate_anchors method and paddle version.
from core.workspace import register
from . import anchor_utils

@register
class GeneartePriorBoxes(object):
    '''
    both for fpn and single layer, single layer need to test
    return (np.array) [num_anchros, 4] [x0, y0, x1, y1]
    '''
    def __init__(self, scale_list=[1.], \
                 aspect_ratio_list=[1.0], \
                 stride_list=[4,8,16,32,64,128], \
                 anchor_size_list=[16,32,64,128,256,512]):
        self.scale_list = scale_list
        self.aspect_ratio_list = aspect_ratio_list
        self.stride_list = stride_list
        self.anchor_size_list = anchor_size_list

    def __call__(self, img_height, img_width):
        final_anchor_list = []

        for idx, stride in enumerate(self.stride_list):
            anchor_list = []
            cur_img_height = img_height
            cur_img_width = img_width
            tmp_stride = stride 

            while tmp_stride != 1:
                tmp_stride = tmp_stride // 2
                cur_img_height = (cur_img_height + 1) // 2
                cur_img_width = (cur_img_width + 1) // 2

            for i in range(cur_img_height):
                for j in range(cur_img_width):
                    for scale in self.scale_list:
                        cx = (j + 0.5) * stride
                        cy = (i + 0.5) * stride
                        side_x = self.anchor_size_list[idx] * scale
                        side_y = self.anchor_size_list[idx] * scale
                        for ratio in self.aspect_ratio_list:
                            anchor_list.append([cx, cy, side_x / math.sqrt(ratio), side_y * math.sqrt(ratio)])

            final_anchor_list.append(anchor_list)
        final_anchor_arr = np.concatenate(final_anchor_list, axis=0)
        normalized_anchor_arr = anchor_utils.normalize_anchor(final_anchor_arr).astype('float32')

        return normalized_anchor_arr

if __name__ == '__main__':
    from wider_util import WiderFace
    import time
    wider_face = WiderFace()
    for img, _, bbox_labels in wider_face:
        np_t0 = time.time()
        #ssd_final_anchor_arr = generate_prior_boxes(settings.resize_height, settings.resize_width)
        ssd_final_anchor_arr = generate_prior_boxes(80, 80)
        import pdb;pdb.set_trace()
        np_t1 = time.time()
        bbox_labels = np.array(bbox_labels)
        bbox_labels = anchor_target(ssd_final_anchor_arr, bbox_labels)
        np_t2 = time.time()
        print ('np_ssd_time gen_anchor: {} anchor_target: {} total_time: {}'.format(np_t1 - np_t0, np_t2 - np_t1, np_t2 - np_t0))
        np_t0 = time.time()
        faster_final_anchor_arr = generate_fpn_anchors_opr(settings.resize_height, settings.resize_width)
        np_t1 = time.time()
        np_t2 = time.time()
        print ('np_faster_time gen_anchor: {} anchor_target: {} total_time: {}'.format(np_t1 - np_t0, np_t2 - np_t1, np_t2 - np_t0))
        #import pdb;pdb.set_trace()

        


