# ******************************************************
# Author       : liuyang
# Last modified:	2020-01-13 20:31
# Email        : gxly1314@gmail.com
# Filename     :	train.py
# Description  : 
# ******************************************************
from __future__ import absolute_import
import sys
import argparse
import numpy as np
import torch
import os
import torch.backends.cudnn as cudnn
from core.workspace import register, create, global_config, load_config
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
from evaluation.evaluate_ap50 import evaluation_ap50
from utils.logger import SimulLogger
import cv2
cv2.setNumThreads(0)




parser = argparse.ArgumentParser(description='Training Details')
parser.add_argument('--batch_size', '-b', default=28, type=int, help='Batch size of all GPUs for training')
parser.add_argument('--num_workers','-n', default=14, type=int, help='Number of workers used in dataloading')
parser.add_argument('--sub_project_name', default=None, type=str, help='sub_project_name.')
parser.add_argument('--config', '-c', default='configs/base_setting/config.yml', type=str, help='config yml.')
parser.add_argument('--resume_iter', '-r', default=None, type=int, help='Resume from checkpoint')

def gen_dir(dir_name_list):
    for dir_name in dir_name_list:
        if not os.path.exists(dir_name):
            os.system('mkdir -p {}'.format(dir_name))

if __name__ == '__main__':
    args = parser.parse_args()
    cfg = load_config(args.config)

    # generate seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed + 10)
    torch.cuda.manual_seed(cfg.seed + 10)

    # generate config related tensorboard, log, snapshot dirs.
    gen_dir_list = []
    config_name = args.config.split('/')[-1].split('.')[-2]
    tensorboard_dir = os.path.join('./tensorboards', config_name) 
    log_dir = os.path.join('./logs', config_name)
    snapshots_dir = os.path.join('./snapshots', config_name)
    gen_dir_list = [snapshots_dir, tensorboard_dir, log_dir]
    gen_dir(gen_dir_list)

    # simultaneous print on terminal and log
    log_file_name = os.path.join(log_dir, 'log.txt')
    sys.stdout = SimulLogger(log_file_name)

    # add tensorboard
    tb_writer = SummaryWriter(log_dir=tensorboard_dir)

    # train data feed
    dataset = create(cfg.train_feed)
    epoch_size = len(dataset) // args.batch_size

    if 'collate_fn' in cfg:
        collate_fn = create(cfg.collate_fn)
        data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                      shuffle=True, pin_memory=True, collate_fn=collate_fn)
    else:
        data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                      shuffle=True, pin_memory=True)

    batch_iterator = iter(data_loader)

    # val data iter
    val_set= create(cfg.validation_set)

    # net
    cfg['phase'] = 'training'
    net = create(cfg.architecture)
    net.cuda()
    parallel_net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    parallel_net = parallel_net.cuda()

    # reusume
    start_iter = 0
    if args.resume_iter is not None:
        model_name = os.path.join(snapshots_dir, 'model_{}000.pth'.format(args.resume_iter))
        print ('Load model from {}'.format(model_name))
        net.load_state_dict(torch.load(model_name))
        start_iter = int(args.resume_iter) * 1000
        print ('Finish load model.')
    # load pretrain_weights
    elif 'pretrain_weights' in cfg:
        net.backbone.load_weights(cfg.pretrain_weights)

    # optimizer
    use_warmup = 'warmup_strategy' in cfg
    if use_warmup:
        warmup_wrapper = create(cfg.warmup_strategy)
    optimizer_wrapper = create(cfg.optimizer)
    optimizer = optimizer_wrapper(parallel_net)

    # epoch_num
    cur_epoch_num = 0

    use_ali_ams = False
    if 'use_ali_ams' in cfg and cfg['use_ali_ams']:
        use_ali_ams = True

    use_hcam = False
    if 'use_hcam' in cfg and cfg['use_hcam']:
        use_hcam = True

    # criterion
    criterion = create(cfg.criterion)
    if use_hcam:
        criterion_1 = create(cfg.criterion_1)
    # train_model
    net.train()
    for iter_idx in range(start_iter, cfg.max_iterations + 1):
        t1 = time.time()
        if use_warmup:
            warmup_wrapper(optimizer, iter_idx)
        try:
            # load train data
            if use_ali_ams:
                images, targets, anchors, bbox_labels_list = next(batch_iterator)
            else:
                images, targets = next(batch_iterator)
        except StopIteration:
            cur_epoch_num += 1
            batch_iterator = iter(data_loader)

        t3 = time.time()
        images = Variable(images.cuda())
        with torch.no_grad():
            targets = Variable(targets.cuda())
            if use_ali_ams:
                bbox_labels_list  = [Variable(bbox_label.cuda()) for bbox_label in bbox_labels_list]
                anchors = Variable(anchors.cuda())
        if use_ali_ams:
            cls, loc = parallel_net(images)
            loss = \
                  criterion(cls, loc, targets, anchors, bbox_labels_list)
            cls_loss, loc_loss, total_loss = loss
            
        elif use_hcam:
            cls, loc, cls_1 = parallel_net(images)
            loss = criterion(cls, loc, targets)
            cls_loss, loc_loss, total_loss, fp_label  = loss
            cls_loss_1 = criterion_1(cls_1, fp_label)
            total_loss += cls_loss_1
        else:
            cls, loc = parallel_net(images)
            loss = criterion(cls, loc, targets)
            cls_loss, loc_loss, total_loss = loss

        optimizer.zero_grad()

        total_loss.backward()
        if 'clip_gradients' in cfg:
            clip_gradients_op = create(cfg.clip_gradients)
            clip_gradients_op(parallel_net.parameters())

        optimizer.step()
        t2 = time.time()

        # add tensorboard
        tb_writer.add_scalar('cls_loss', cls_loss, iter_idx)
        tb_writer.add_scalar('loc_loss', loc_loss, iter_idx)
        tb_writer.add_scalar('total_loss', total_loss, iter_idx)
        if len(loss) == 5:
            tb_writer.add_scalar('cls_loss_2', cls_loss_2, iter_idx)

        if iter_idx % cfg.log_smooth_window == 0:
            print ('Epoch: {}, Iter: {}, time: {:.4f}s, data aug time: {:.4f}s'.format(cur_epoch_num, iter_idx, t2 - t1, t3 - t1))
            if use_hcam:
                print('Loss conf: {:.4f} Loss loc: {:.4f} Loss_hcam: {:.4f}'.format(cls_loss.data, loc_loss.data, cls_loss_1.data))
            else:
                print('Loss conf: {:.4f} Loss loc: {:.4f}'.format(cls_loss.data, loc_loss.data))
            print('lr: {:.4f}'.format(optimizer.param_groups[0]['lr']))

        if iter_idx % cfg.snapshot_iter == 0 and iter_idx != start_iter:
            if iter_idx == 0:
                save_model_name = os.path.join(snapshots_dir, 'model_0000.pth')
            else:
                save_model_name = os.path.join(snapshots_dir, 'model_{}.pth'.format(iter_idx))
            print('Iter: {}, Saving model in {}.' .format(iter_idx, save_model_name))

            torch.save(net.state_dict(), save_model_name)

        if iter_idx in cfg.eval_iter_list:
            # add singgle scale eval_net
            os.system('CUDA_VISIBLE_DEVICES={} python test_single.py -n {} -c {}' \
                      .format(os.environ["CUDA_VISIBLE_DEVICES"][0], int(iter_idx / 1000), args.config))
    tb_writer.close()

