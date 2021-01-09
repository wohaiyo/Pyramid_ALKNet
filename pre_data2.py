from __future__ import print_function
import numpy as np
import os
import scipy.misc as misc
import config as cfg
import random
import cv2

IMAGE_HEIGHT = cfg.IMAGE_HEIGHT
IMAGE_WIDTH = cfg.IMAGE_WIDTH

gr_shape = cfg.IMAGE_HEIGHT


def data_crop_test_output(session, gr_data, logits, image, mean, std, mean_rgb, crop_size_h, crop_size_w, stride):
    image_h = image.shape[0]
    image_w = image.shape[1]
    pad_h = 0
    pad_w = 0
    if image_h >= crop_size_h and image_w >= crop_size_w:
        image_pad = image
    else:
        if image_h < crop_size_h:
            pad_h = crop_size_h - image_h
        if image_w < crop_size_w:
            pad_w = crop_size_w - image_w
        image_pad = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=0)
    image_pad = np.asarray(image_pad, dtype='float32')

    # image_pad = image_pad * (1.0 / 255)           # sub mean and div std
    # image_pad = (image_pad - mean) / std
    image_pad = image_pad - mean_rgb                # sub rgb mean

    image_crop_batch = []
    x_start = [x for x in range(0, image_pad.shape[0] - crop_size_h + 1, stride)]
    y_start = [y for y in range(0, image_pad.shape[1] - crop_size_w + 1, stride)]
    if (image_pad.shape[0] - crop_size_h) % stride != 0:
        x_start.append(image_pad.shape[0] - crop_size_h)
    if (image_pad.shape[1] - crop_size_w) % stride != 0:
        y_start.append(image_pad.shape[1] - crop_size_w)
    for x in x_start:
        for y in y_start:
            image_crop_batch.append(image_pad[x:x + crop_size_h, y:y + crop_size_w])

    logit = []
    for crop_batch in image_crop_batch:
        lo = session.run(
        logits,
        feed_dict={
            gr_data: [crop_batch]
        })
        logit.append(lo[0])
    # logit = session.run(
    #     logits,
    #     feed_dict={
    #         gr_data: image_crop_batch, keep_pro: 1.0
    #     })

    num_class = cfg.NUM_OF_CLASSESS
    score_map = np.zeros([image_pad.shape[0], image_pad.shape[1], num_class], dtype='float32')
    count = np.zeros([image_pad.shape[0], image_pad.shape[1], num_class], dtype='float32')
    crop_index = 0
    for x in x_start:
        for y in y_start:
            crop_logits = logit[crop_index]
            score_map[x:x + crop_logits.shape[0], y:y + crop_logits.shape[1], :] += crop_logits
            count[x:x + crop_logits.shape[0], y:y + crop_logits.shape[1], :] += 1
            crop_index += 1

    score_map = score_map[:image_h, :image_w] / count[:image_h, :image_w]
    return score_map


# if __name__ == "__main__":
#
#     img = cv2.imread('data/ecp/train/img/monge_5.jpg')
#     image = img_as_float(img)
#     bright = 0.5
#     contrast = bright
#     gam1 = exposure.adjust_gamma(image, bright)
#     gam2 = exposure.adjust_gamma(gam1, contrast)
#     gam2 = img_as_ubyte(gam2)
#     cv2.imshow('img.jpg', gam2)
#     cv2.waitKey(0)
#
#     im_list = getList()
#     im, an, m, b, e = getTraindata(im_list, 4)
#     im_list = getValList()
#     im, an, m, b, e = getValdata(im_list, 4)
#     print('ok')