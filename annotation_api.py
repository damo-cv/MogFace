from __future__ import absolute_import
from fastapi import FastAPI, Request
from pydantic import BaseModel
import boto3
import numpy as np
import torch
import os
import cv2
from core.workspace import create, load_config
from torch.autograd import Variable
from utils.nms.nms_wrapper import nms
from data import anchor_utils
from data.preprocess import BasePreprocess
from data.data_aug_settings import DataAugSettings
from data.anchors_opr import GeneartePriorBoxes


app = FastAPI()

class Video(BaseModel):
    bucket_name: str
    video_key: str

# Create an S3 client
s3 = boto3.client('s3')

#################################
# Parameters
#################################

ANNOTATION_BUCKET = "veesion-blurring-annotations"

num_iter = 140 # number of iteration for test.
nms_th = 0.4 # nms threshold.
pre_nms_top_k = 5000 # number of max score image.
score_th = 0.01 # score threshold.
max_bbox_per_img = 750 # max number of det bbox.
scale_weight = 10 # to differentiate the gap between large and small scale..
max_img_shrink = 1.1 # constrain the max shrink of img.
vote_th = 0.5 # bbox vote threshold
config = 'configs/mogface/MogFace.yml' # config yml.
sub_project_name = None # sub_project_name.
test_min_scale = 0 # the min scale of det bbox
flip_ratio = 1.4
test_hard = 0
videos_path = "/workspace/videos/"

#################################
# Functions
#################################

def detect_face(image, shrink, anchors_function):
    x = image
    if shrink != 1:
        x = cv2.resize(image, None, None, fx=shrink, fy=shrink,
                       interpolation=cv2.INTER_LINEAR)


    width = x.shape[1]
    height = x.shape[0]

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = Variable(x.cuda(), volatile=True)

    out = net(x)

    anchors = anchor_utils.transform_anchor((anchors_function(height, width)))
    anchors = torch.FloatTensor(anchors).cuda()
    decode_bbox = anchor_utils.decode(out[1].squeeze(0), anchors)
    boxes = decode_bbox
    scores = out[0].squeeze(0)

    top_k = pre_nms_top_k
    v, idx = scores[:, 0].sort(0)
    idx = idx[-top_k:]
    boxes = boxes[idx]
    scores = scores[idx]

    boxes = boxes.cpu().numpy()
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    boxes[:, 0] /= shrink
    boxes[:, 1] /= shrink
    boxes[:, 2] = boxes[:, 0] + w / shrink - 1
    boxes[:, 3] = boxes[:, 1] + h / shrink - 1
    if int(test_min_scale) != 0:
        boxes_area = (boxes[:, 3] - boxes[:, 1] + 1) * \
            (boxes[:, 2] - boxes[:, 0] + 1) / (shrink * shrink)
        boxes = boxes[boxes_area > int(test_min_scale)**2]
        scores = scores[boxes_area > int(test_min_scale)**2]

    scores = scores.cpu().numpy()

    inds = np.where(scores[:, 0] > score_th)[0]
    if len(inds) == 0:
        det = np.empty([0, 5], dtype=np.float32)
        return det
    c_bboxes = boxes[inds]
    c_scores = scores[inds, 0]
    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
        np.float32, copy=False)

    keep = nms(c_dets, nms_th)
    c_dets = c_dets[keep, :]

    if max_bbox_per_img > 0:
        image_scores = c_dets[:, -1]
        if len(image_scores) > max_bbox_per_img:
            image_thresh = np.sort(image_scores)[-max_bbox_per_img]
            keep = np.where(c_dets[:, -1] >= image_thresh)[0]
            c_dets = c_dets[keep, :]
    return c_dets


def multi_scale_test(image, max_im_shrink, anchors_function):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(image, st, anchors_function=anchors_function)
    if max_im_shrink > 0.75:
        det_s = np.row_stack(
            (det_s, detect_face(image, 0.75, anchors_function=anchors_function)))
    if scale_weight == -1:
        index = np.where(np.maximum(
            det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    else:
        index = np.where(((det_s[:, 2] - det_s[:, 0])
                         * (det_s[:, 3] - det_s[:, 1])) > 2000)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (
        st + max_im_shrink) / 2
    det_b = detect_face(image, bt, anchors_function=anchors_function)
    if scale_weight == -1:
        index = np.where(np.minimum(
            det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
    else:
        index = np.where(((det_b[:, 2] - det_b[:, 0]) *
                         (det_b[:, 3] - det_b[:, 1])) < scale_weight * 600)[0]
    det_b = det_b[index, :]

    # enlarge small iamge x times for small face
    if max_im_shrink > 1.5:
        det_tmp = detect_face(image, 1.5, anchors_function=anchors_function)
        if scale_weight == -1:
            index = np.where(np.minimum(
                det_tmp[:, 2] - det_tmp[:, 0] + 1, det_tmp[:, 3] - det_tmp[:, 1] + 1) < 100)[0]
        else:
            index = np.where(((det_tmp[:, 2] - det_tmp[:, 0]) * (
                det_tmp[:, 3] - det_tmp[:, 1])) < scale_weight * 800)[0]
        det_tmp = det_tmp[index, :]
        det_b = np.row_stack((det_b, det_tmp))

    if max_im_shrink > 2:
        det_tmp = detect_face(image, max_im_shrink,
                              anchors_function=anchors_function)
        if scale_weight == -1:
            index = np.where(np.minimum(
                det_tmp[:, 2] - det_tmp[:, 0] + 1, det_tmp[:, 3] - det_tmp[:, 1] + 1) < 100)[0]
        else:
            index = np.where(((det_tmp[:, 2] - det_tmp[:, 0]) * (
                det_tmp[:, 3] - det_tmp[:, 1])) < scale_weight * 500)[0]
        det_tmp = det_tmp[index, :]
        det_b = np.row_stack((det_b, det_tmp))

    return det_s, det_b


def multi_scale_test_pyramid(image, max_shrink, anchors_function):
    # shrink detecting and shrink only detect big face
    det_b = detect_face(image, 0.25, anchors_function=anchors_function)
    if scale_weight == -1:
        index = np.where(
            np.maximum(det_b[:, 2] - det_b[:, 0] + 1,
                       det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
    else:
        index = np.where(((det_b[:, 2] - det_b[:, 0])
                         * (det_b[:, 3] - det_b[:, 1])) > 2000)[0]
    det_b = det_b[index, :]

    st = [1.25, 1.75, 2.25]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            det_temp = detect_face(
                image, st[i], anchors_function=anchors_function)
            # enlarge only detect small face
            if i == 0:
                if scale_weight == -1:
                    index = np.where(
                        np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                                   det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                else:
                    index = np.where(((det_temp[:, 2] - det_temp[:, 0]) * (
                        det_temp[:, 3] - det_temp[:, 1])) < scale_weight * 2000)[0]
                det_temp = det_temp[index, :]
            if i == 1:
                if scale_weight == -1:
                    index = np.where(
                        np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                                   det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                else:
                    index = np.where(((det_temp[:, 2] - det_temp[:, 0]) * (
                        det_temp[:, 3] - det_temp[:, 1])) < scale_weight * 1000)[0]
                det_temp = det_temp[index, :]
            if i == 2:
                if scale_weight == -1:
                    index = np.where(
                        np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                                   det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                else:
                    index = np.where(((det_temp[:, 2] - det_temp[:, 0]) * (
                        det_temp[:, 3] - det_temp[:, 1])) < scale_weight * 600)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))

    return det_b


def flip_test(image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink, anchors_function=anchors_function)

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
    det[:, :4] = np.round(det[:, :4])
    dets = []
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
        union[union <= 0] = 1
        o = inter / union
        o[0] = 1

        # get needed merge det and delete these det
        merge_index = np.where(o >= vote_th)[0]
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
        det_accu_sum[:, 0:4] = np.sum(
            det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    if not len(dets):
        return dets
    if dets.shape[0] > 750:
        dets = dets[0:750, :]
    return dets


def write_to_txt(f, det, height, width, image_id):
    if not len(det):
        return
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
        f.write('{} {:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.
                format(image_id, round(xmin), round(ymin), round(xmax - xmin + 1), round(ymax - ymin + 1), score))


def gen_soft_link_dir(dir_name_list):
    for dir_name in dir_name_list:
        cur_dir_name = dir_name.split('/')[-1]
        if os.path.exists(cur_dir_name):
            os.system('rm -rf ./{}'.format(cur_dir_name))
        if not os.path.exists(dir_name):
            raise ValueError(
                'Cannot create soft link, {} does not exist'.format(dir_name))
        os.system('ln -s {} ./{}'.format(dir_name, cur_dir_name))


def gen_dir(dir_name_list):
    for dir_name in dir_name_list:
        if not os.path.exists(dir_name):
            os.system('mkdir -p {}'.format(dir_name))


def generate_annotations_file(video_path, annotations_file):
    video_reader = cv2.VideoCapture(video_path)
    is_frame_valid, img = video_reader.read()
    if not is_frame_valid:
        print(f"Error: video {video_path} not valid")
        return
    image_id = 0
    while is_frame_valid:
        img = preprocess_fn(img, phase='validation')
        with torch.no_grad():
            # the max size of input image for caffe
            max_im_shrink = (0x7fffffff / 200.0 /
                                (img.shape[0] * img.shape[1])) ** 0.5
            max_im_shrink = max_img_shrink if max_im_shrink > max_img_shrink else max_im_shrink
            shrink = max_im_shrink if max_im_shrink < 1 else 1
            det0 = detect_face(
                img, shrink, anchors_function=anchors_function)  # origin test
            det1 = flip_test(img, shrink)    # flip test

        det = np.row_stack((det0, det1))
        dets = bbox_vote(det)
        mode = "a" if os.path.exists(annotations_file) else "w"

        with open(annotations_file, mode) as f:
            write_to_txt(f, dets, img.shape[0], img.shape[1], image_id)
        is_frame_valid, img = video_reader.read()
        image_id += 1

#################################
# Initialization
#################################

# generate det_info and det_result
cfg = load_config(config)
cfg['phase'] = 'test'

config_name = config.split('/')[-1].split('.')[-2]
snapshots_dir = os.path.join('./snapshots', config_name)

det_info_dir = os.path.join('./det_info', config_name)

det_result_dir = os.path.join('./det_result', config_name)

save_info_dir_name = 'ss_' + str(num_iter) + '_nmsth_' + str(nms_th) + \
    '_scoreth_' + str(score_th)

abs_save_dir = os.path.join(det_info_dir, save_info_dir_name)
det_result_txt = os.path.join(det_result_dir, 'result.txt')

gen_dir_list = [abs_save_dir, det_result_dir]
gen_dir(gen_dir_list)

# create net and val_set
net = create(cfg.architecture)
model_name = os.path.join(
    snapshots_dir, 'model_{}000.pth'.format(num_iter))
print('Load model from {}'.format(model_name))
net.load_state_dict(torch.load(model_name))
net.cuda()
net.eval()
print('Finish load model.')

anchors_function = GeneartePriorBoxes(
    scale_list=[0.68], aspect_ratio_list=[1.0], stride_list=[4, 8, 16, 32, 64, 128], anchor_size_list=[16, 32, 64, 128, 256, 512])

# generate predict bbox
preprocess_fn = BasePreprocess(
    data_aug_settings=DataAugSettings(), normalize_pixel=True, use_rgb=True, img_mean=[0.485, 0.456, 0.406], img_std=[0.229, 0.224, 0.225])

#################################
# API
#################################

@app.post("/cloud")
async def detect_faces_in_cloud(video: Video, request: Request):
    video_directory = "/tmp/"
    video_name = video.video_key.split("/")[-1]
    video_path = os.path.join(video_directory, video_name)
    # Download alert video
    try:
        s3.download_file(video.bucket_name, video.video_key, video_path)
        print(f"Object '{video.video_key}' downloaded to '{video_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    
    # Generate .txt
    output_directory = "annotations"
    annotations_file_name = ".".join(video_name.split(".")[:-1])+".txt"
    annotations_file = os.path.join(output_directory, annotations_file_name)
    print(f"Generating annotation file for video {video.video_key} in bucket {video.bucket_name}")
    generate_annotations_file(video_path, annotations_file)
    print(f"Annotation file generated for video {video.video_key} in bucket {video.bucket_name}")

    # Upload
    annotation_key = f"{video.video_key}.txt"
    s3.upload_file(Filename=annotations_file, Bucket=ANNOTATION_BUCKET, Key=annotation_key, ExtraArgs={"Tagging": f"bucket_name={video.bucket_name}"})
    print(f"Annotation file {annotation_key} uploaded in bucket {ANNOTATION_BUCKET}")

    # Cleaning
    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(annotations_file):
        os.remove(annotations_file)

    return {"msg": f"Annotation file {annotation_key} uploaded in bucket {ANNOTATION_BUCKET}"}
