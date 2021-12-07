#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw


def draw_bbox(image, bbox):
    """
    Draw one bounding box on image.
    Args:
        image (PIL.Image): a PIL Image object.
        bbox (np.array|list|tuple): (xmin, ymin, xmax, ymax).
    """
    draw = ImageDraw.Draw(image)
    xmin, ymin, xmax, ymax = box
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top),
         (left, top)],
        width=4,
        fill='red')


def draw_bboxes(img, bboxes, labels=None,  output_dir='../tmp_img', save_img_name=None):
    """
    Draw bounding boxes on image.
    
    Args:
        img (np.ndarray or string): input image or image path.
        bboxes (np.array): bounding boxes.
        labels (list of string): the label names of bboxes.
        output_dir (string): output directory.
    """
    if labels is not None:
        assert len(bboxes) == len(labels), 'bboxes and labels should be consistent.'
    if type(img) != np.ndarray:
        if save_img_name is None:
            save_img_name = img.split('/')[-1]
        img = cv2.imread(img)
    else:
        if save_img_name is None:
            save_img_name = '{}.jpg'.format(np.random.uniform(0,1))

    for i in range(len(bboxes)):
        xmin, ymin, xmax, ymax = bboxes[i]
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0), 4)
        if labels is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img, str(labels[i]), (int(xmin), int(ymin)), font, 0.8, (0, 0, 255), 2) 

    save_img_name = os.path.join(output_dir, save_img_name)
    cv2.imwrite(save_img_name, img)
    print("The image with bbox is saved as {}".format(save_img_name))
