from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import glob
import random
import scipy.misc as misc
import config as cfg
from pre_data2 import data_crop_test_output
import cv2



def evaluate_accuracy(fc8s_logits, sess, image, img_name):


    logits = fc8s_logits

    crop_size_h = cfg.IMAGE_HEIGHT # 480 512 500 224
    crop_size_w = cfg.IMAGE_WIDTH
    stride = int(crop_size_w / 1)
    mean = [0.485, 0.456, 0.406]  # rgb
    std = [0.229, 0.224, 0.225]
    # mean_rgb = [123.68, 116.78, 103.94]  # rgb mean subtract

    mean_bgr = [103.94, 116.78, 123.68]


    if not os.path.exists(cfg.save_dir + 'output'):
        os.mkdir(cfg.save_dir + 'output')
    f = open(cfg.save_dir + 'output/result.txt', 'w')
    total_acc = 0
    total_acc_list = []
    total_acc_cls = []
    total_pre_cls = []
    total_rec_cls = []
    total_f1_cls = []

    total_tp_num = []
    total_all_num = []



    valid_images = [cv2.imread(img_name)]
    img_ori = valid_images[0]
    h_ori, w_ori, _ = img_ori.shape


    # scs = [0.75, 1.0, 1.25, 1.5]
    # scs = [0.5, 0.75,  1.0,  1.5, 2.0]
    # scs = [0.75, 1.25, 1.75]
    scs = [1.25]    # for ecp
    # scs = [0.5]   # for ruemonge


    maps = []
    for sc in scs:
        img = cv2.resize(img_ori, (int(float(w_ori) * sc), int(float(h_ori) * sc)), interpolation=cv2.INTER_LINEAR)

        score_map = data_crop_test_output(sess, image, logits, img, mean, std, mean_bgr, crop_size_h,
                                          crop_size_w, stride)
        score_map = cv2.resize(score_map, (w_ori, h_ori), interpolation=cv2.INTER_LINEAR)
        maps.append(score_map)
    score_map = np.mean(np.stack(maps), axis=0)

    # maps2 = []
    # for sc in scs:
    #     img2 = cv2.resize(img_ori, (int(float(w_ori) * sc), int(float(h_ori) * sc)), interpolation=cv2.INTER_LINEAR)
    #     img2 = cv2.flip(img2, 1)
    #     score_map2 = data_crop_test_output(sess, image, logits, img2, mean, std, mean_bgr, crop_size_h,
    #                                       crop_size_w, stride)
    #     score_map2 = cv2.resize(score_map2, (w_ori, h_ori), interpolation=cv2.INTER_LINEAR)
    #     maps2.append(score_map2)
    # score_map2 = np.mean(np.stack(maps2), axis=0)
    # score_map2 = cv2.flip(score_map2, 1)
    # score_map = (score_map + score_map2) / 2

    pred_label = np.argmax(score_map, 2)
    pred_label = np.asarray(pred_label, dtype='uint8')
    pred = pred_label[:, :, np.newaxis]

    return pred
