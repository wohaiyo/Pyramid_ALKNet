from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import glob
import random
import scipy.misc as misc
import config as cfg
import time
import csv
from tensorflow.python import pywrap_tensorflow
from facade_network import inference_Pyramid_ALKNet, inference_Pyramid_ALKNet2, inference_py_alk_512,\
    inference_deeplabv3_plus_16, inference_resnet50, inference_pspnet, inference_danet, \
    inference_deeplabv3_plus_16, inference_pspnet, inference_resnet50, \
    inference_Pyramid_ALKNet_2feat, inference_Pyramid_ALKNet_3feat, \
    inference_Pyramid_ALKNet_two_layer, inference_Pyramid_ALKNet_one_layer, \
    inference_Pyramid_ALKNet_k13, inference_Pyramid_ALKNet_k11, inference_deeplabv3_plus_16_init

from utils import pred_vision, eval_img2, eval_fscore, pred_vision_path
from pre_data2 import data_crop_test_output
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.use_gpu

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string('mode', "self_att", "Mode train/ test/ visualize")  # visualize  train mul_eval

IMAGE_SIZE = None
class_names_ignore_background = []                                      # Ignore backgournd label
for i in range(1, len(cfg.class_names)):
    class_names_ignore_background.append(cfg.class_names[i])
cfg.class_names = class_names_ignore_background


def get_variables_in_checkpoint_file(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")

def get_variables_to_restore(variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:

        if v.name.split(':')[0] in var_keep_dic:
            print('Variables restored: %s' % v.name)
            variables_to_restore.append(v)

    return variables_to_restore

def fast_hist(a, b, n):         # a: gt, b: pred
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def getRects(score_map, class_num):
    score_map = np.argmax(score_map, 2)
    score_copy = np.zeros(score_map.shape)
    score_copy[(score_map == class_num)] = 1
    score_copy = np.array(score_copy, np.uint8)
    _, cnts, _ = cv2.findContours(score_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for cors in cnts:
        cors = np.squeeze(cors, 1)
        if max(cors[:, 0]) - min(cors[:, 0]) > 5 and max(cors[:, 1]) - min(cors[:, 1]) > 5:
            # x1.append(min(cors[:, 0]))
            # x2.append(max(cors[:, 0]))
            # y1.append(min(cors[:, 1]))
            # y2.append(max(cors[:, 1]))

            rect = cv2.boundingRect(cors)
            x1.append(rect[0])
            y1.append(rect[1])
            x2.append(rect[0]+rect[2])
            y2.append(rect[1]+rect[3])

    x1 = np.array(x1)[:, np.newaxis]
    x2 = np.array(x2)[:, np.newaxis]
    y1 = np.array(y1)[:, np.newaxis]
    y2 = np.array(y2)[:, np.newaxis]
    desCor = np.concatenate([x1, y1, x2, y2], axis=1)
    desCor = np.array(desCor, np.int32)

    return desCor



def main(argv=None):
    image = tf.placeholder(tf.float32, shape=[1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3], name="input_image")

    pred_annotation,  fc8s_logits = inference_danet(image, is_training=False)
    # pred_annotation, fc8s_logits, _ = inference_pspnet(image, is_training=False)
    # pred_annotation, fc8s_logits = inference_Pyramid_ALKNet(image, is_training=False)
    # pred_annotation, fc8s_logits = inference_deeplabv3_plus_16(image, is_training=False)

    f_test = open(cfg.test_data_list, 'r')
    img_list = []
    label_list = []
    for line in f_test:
        try:
            image_name, label = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            image_name = label = line.strip("\n")
        img_list.append(cfg.test_data_dir + image_name)
        label_list.append(cfg.test_data_dir + label)
    f_test.close()

    logits = tf.nn.softmax(fc8s_logits)
    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()


    is_best = False     # Use the best model to evaluate
    epo = 0
    files = os.path.join(cfg.save_dir + 'model.ckpt-*.index')
    sfile = glob.glob(files)
    if len(sfile) >= 0:
        sess.run(tf.global_variables_initializer())
        if is_best:
            model = cfg.save_dir + 'best.ckpt'
        else:
            sfile = glob.glob(files)
            steps = []
            for s in sfile:
                part = s.split('.')
                step = int(part[1].split('-')[1])
                steps.append(step)
            epo = max(steps)

            # Which model to eval
            model = cfg.save_dir + 'model.ckpt-' + str(epo)

        print('\nRestoring weights from: ' + model)
        saver.restore(sess, model)
        print('End Restore')
    else:
        # restore from pre-train on imagenet or pre-trained
        variables = tf.global_variables()
        sess.run(tf.variables_initializer(variables, name='init'))
        if os.path.exists(cfg.pre_trained_model) or os.path.exists(cfg.pre_trained_model + '.index'):
            var_keep_dic = get_variables_in_checkpoint_file(cfg.pre_trained_model)
            # Get the variables to restore, ignoring the variables to fix
            variables_to_restore = get_variables_to_restore(variables, var_keep_dic)
            # var_to_restore = [val for val in variables if 'conv1' in val.name or 'conv2' in val.name or
            #                   'conv3' in val.name or 'conv4' in val.name or 'conv5' in val.name]
            if len(variables_to_restore) > 0:
                restorer = tf.train.Saver(variables_to_restore)
                restorer.restore(sess, cfg.pre_trained_model)
                print('Vgg model pre-train Loaded')
            else:
                print('Model inited random.')
        else:
            print('Model inited random.')


    if FLAGS.mode == "train":
        print('Start train etrims...')

    elif FLAGS.mode == "eval":              # visual feature
        if not os.path.exists(cfg.save_dir + 'output'):
            os.mkdir(cfg.save_dir + 'output')
        if not os.path.exists(cfg.save_dir + 'output_feat'):
            os.mkdir(cfg.save_dir + 'output_feat')

        total_acc = 0
        total_acc_cls = []
        import cv2
        for item in range(len(img_list)):
            valid_images = [cv2.imread(img_list[item]) - np.array([103.94, 116.78, 123.68], dtype=np.float32)]
            valid_annotations = [np.expand_dims(misc.imread(label_list[item]), axis=2)]
            im_name = img_list[item].split('/')[-1].split('.')[0]

            valid_images2 = [cv2.resize(valid_images[0], (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))]
            cv2.imwrite('.jpg', valid_images2[0] + np.array([103.94, 116.78, 123.68], dtype=np.float32))
            edge_mask,feature = sess.run([logits, feat], feed_dict={image: valid_images2})#, edge: eval_edge2})

            score_map = cv2.resize(edge_mask[0], (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
            valid_anno = cv2.resize(valid_annotations[0], (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT),
                                              interpolation=cv2.INTER_NEAREST)
            valid_anno_8s = cv2.resize(valid_annotations[0], (int(cfg.IMAGE_WIDTH/8), int(cfg.IMAGE_HEIGHT/8)),
                                    interpolation=cv2.INTER_NEAREST)
            pred_label = np.argmax(score_map, 2)
            pred_label = np.asarray(pred_label, dtype='uint8')
            pred = [pred_label[:, :, np.newaxis]]

            pred_vision(pred[0], im_name, cfg.dataset)
            pred_vision(valid_anno, im_name + '_gt', cfg.dataset)


            from utils import PCA_compress
            from utils import t_sne_compress
            from utils import visual_2d

            # feature = PCA_compress(feature)
            # 1
            feature1 = t_sne_compress(feature, 1)
            feature1_img = cv2.resize(feature1[0], (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
            feature1_img = np.array(feature1_img * 255, np.uint8)
            cv2.imwrite(cfg.save_dir + 'output_feat/' + im_name + '_feature1.jpg', feature1_img)
            print('feature1 done!')
            # 2
            feature2 = t_sne_compress(feature, 2)
            visual_2d(feature2[0], valid_anno_8s, cfg.save_dir + 'output_feat/' + im_name)
            print('feature2 done!')
            # 3
            feature3 = t_sne_compress(feature, 3)
            feature3_img = cv2.resize(feature3[0], (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
            feature3_img = np.array(feature3_img * 255, np.uint8)
            cv2.imwrite(cfg.save_dir + 'output_feat/' + im_name + '_feature3.jpg', feature3_img)
            print('feature3 done!')

    elif FLAGS.mode == "self_att":
        print('----- Show the attention map of pixel -----')
        import cv2
        print('Input size: ' + str([cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH]))

        for item in range(len(img_list)):
            ori_img = cv2.imread(img_list[item])
            ori_img_h, ori_img_w = ori_img.shape[0], ori_img.shape[1]

            im_name = img_list[item].split('/')[-1]

            valid_images = cv2.resize(ori_img, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
            valid_images = valid_images - cfg.IMG_MEAN

            # Run
            score_map, gamma_, attention_ = sess.run([logits, gamma, attention], feed_dict={image: [valid_images]})

            score_map = cv2.resize(score_map[0], (ori_img_w, ori_img_h))

            pred_label = np.argmax(score_map, 2)

            # Save to path
            if not os.path.exists(cfg.save_dir + 'output'):
                os.mkdir(cfg.save_dir + 'output')
            save_path = cfg.save_dir + 'output'
            save_name = os.path.join(save_path, im_name)
            # pred_vision(pred_label, save_name)

            print('Gamma: ' + str(gamma_[0]))
            attention_ = np.transpose(attention_[0], (1, 0))
            row = attention_[2252]
            row = np.reshape(row, [cfg.IMAGE_HEIGHT // 8, cfg.IMAGE_WIDTH // 8])
            row = np.expand_dims(row, 2)
            row = cv2.resize(row, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
            pred = row
            pred_new = np.array(pred * 255, np.uint8)
            cv2.imwrite(save_name, pred_new)
            print(save_name + ' is saved.')

            print('image ' + str(item))

    elif FLAGS.mode == "test_time": # ka 0: 0.06800829569498698  ka 1: 0.061087481180826825
        import cv2
        time_count = 0
        time_total = 0
        for i, item in enumerate(range(len(img_list))):
            valid_images = cv2.imread(img_list[item]) - np.array([103.94, 116.78, 123.68], dtype=np.float32)
            valid_images2 = [cv2.resize(valid_images, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))]

            time_1 = time.time()
            re= sess.run(logits, feed_dict={image: valid_images2})  # , edge: eval_edge2})
            time_2 = time.time()
            print('time: ' + str(time_2 - time_1))
            if i > 0:
                time_count += 1
                time_total += (time_2 - time_1)
        print('Average: ' + str(time_total / time_count))
    elif FLAGS.mode == "demo":
        print('---------Start demo -------------')
        crop_size_h = cfg.IMAGE_HEIGHT
        crop_size_w = cfg.IMAGE_WIDTH
        print('crop size: ' + str(crop_size_h))
        stride = int(crop_size_w / 3)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean_bgr = [103.94, 116.78, 123.68]

        save_path = './demo/parsing_results/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img_list = glob.glob('./demo/rgb_images/*.jpg')

        import cv2
        for item in range(len(img_list)):
            valid_images = [cv2.imread(img_list[item])]
            im_name = img_list[item].split('/')[-1].split('.')[0]
            img_ori = valid_images[0]
            h_ori, w_ori, _ = img_ori.shape

            if 'ecp' in cfg.dataset:
                scs = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            elif 'RueMonge' in cfg.dataset:
                scs = [0.45, 0.6, 0.75]

            maps = []
            for sc in scs:
                img = cv2.resize(img_ori, (int(float(w_ori) * sc), int(float(h_ori) * sc)),
                                 interpolation=cv2.INTER_LINEAR)
                score_map = data_crop_test_output(sess, image, logits, img, mean, std, mean_bgr, crop_size_h,
                                                  crop_size_w, stride)
                score_map = cv2.resize(score_map, (w_ori, h_ori), interpolation=cv2.INTER_LINEAR)
                maps.append(score_map)
            score_map = np.mean(np.stack(maps), axis=0)

            maps2 = []
            for sc in scs:
                img2 = cv2.resize(img_ori, (int(float(w_ori) * sc), int(float(h_ori) * sc)),
                                  interpolation=cv2.INTER_LINEAR)
                img2 = cv2.flip(img2, 1)
                score_map2 = data_crop_test_output(sess, image, logits, img2, mean, std, mean_bgr, crop_size_h,
                                                   crop_size_w, stride)
                score_map2 = cv2.resize(score_map2, (w_ori, h_ori), interpolation=cv2.INTER_LINEAR)
                maps2.append(score_map2)
            score_map2 = np.mean(np.stack(maps2), axis=0)
            score_map2 = cv2.flip(score_map2, 1)
            score_map = (score_map + score_map2) / 2

            pred_label = np.argmax(score_map, 2)

            pred_label = np.asarray(pred_label, dtype='uint8')
            pred = [pred_label[:, :, np.newaxis]]

            save_name = save_path + im_name + '.png'
            pred_vision_path(pred[0], save_name, cfg.dataset)

            print('image ' + str(item))
    elif FLAGS.mode == "test_img":
        print('---------Start test img-------------')
        crop_size_h = cfg.IMAGE_HEIGHT
        crop_size_w = cfg.IMAGE_WIDTH
        print('crop size: ' + str(crop_size_h))
        stride = int(crop_size_w / 3)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean_bgr = [103.94, 116.78, 123.68]

        if not os.path.exists(cfg.save_dir + 'output'):
            os.mkdir(cfg.save_dir + 'output')

        test_imgs_path = '/media/ilab/Storage 2/building_warped/'
        # test image path
        img_list = glob.glob(test_imgs_path + '*.jpg') + glob.glob(test_imgs_path + '*.png')

        import cv2
        for item in range(len(img_list)):
            valid_images = [cv2.imread(img_list[item])]
            im_name = img_list[item].split('/')[-1].split('.')[0]
            img_ori = valid_images[0]
            h_ori, w_ori, _ = img_ori.shape

            scs = [0.3, 0.4, 0.5, 0.6, 0.7]
            if 'ecp' in cfg.dataset:
                scs = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
                scs = [0.3, 0.4, 0.5, 0.6, 0.7]
            elif 'RueMonge' in cfg.dataset:
                scs = [0.45, 0.6, 0.75]
                scs = [0.32, 0.5, 0.75] # test building

            maps = []
            for sc in scs:
                img = cv2.resize(img_ori, (int(float(w_ori) * sc), int(float(h_ori) * sc)),
                                 interpolation=cv2.INTER_LINEAR)
                score_map = data_crop_test_output(sess, image, logits, img, mean, std, mean_bgr, crop_size_h,
                                                  crop_size_w, stride)
                score_map = cv2.resize(score_map, (w_ori, h_ori), interpolation=cv2.INTER_LINEAR)
                maps.append(score_map)
            score_map = np.mean(np.stack(maps), axis=0)

            maps2 = []
            for sc in scs:
                img2 = cv2.resize(img_ori, (int(float(w_ori) * sc), int(float(h_ori) * sc)),
                                  interpolation=cv2.INTER_LINEAR)
                img2 = cv2.flip(img2, 1)
                score_map2 = data_crop_test_output(sess, image, logits, img2, mean, std, mean_bgr, crop_size_h,
                                                   crop_size_w, stride)
                score_map2 = cv2.resize(score_map2, (w_ori, h_ori), interpolation=cv2.INTER_LINEAR)
                maps2.append(score_map2)
            score_map2 = np.mean(np.stack(maps2), axis=0)
            score_map2 = cv2.flip(score_map2, 1)
            score_map = (score_map + score_map2) / 2

            # save score map for crf
            if False:
                if not os.path.exists(cfg.save_dir + '/score_map/'):
                    os.makedirs(cfg.save_dir + '/score_map/')
                np.save(cfg.save_dir + '/score_map/' + im_name + '.npy', score_map)

            pred_label = np.argmax(score_map, 2)

            pred_label = np.asarray(pred_label, dtype='uint8')
            pred_label = pred_label[:, :, np.newaxis]
            pred_label_copy = pred_label.copy()
            pred_label[pred_label_copy == 2] = 1    # window
            pred_label[pred_label_copy == 1] = 2    # wall
            pred_label[pred_label_copy == 4] = 1    # balcony
            pred_label[pred_label_copy == 3] = 4    # door


            pred = [pred_label]
            pred_vision(pred[0], im_name, 'ecp')
            # pred_vision(valid_annotations[0], im_name + '_gt', cfg.dataset)
            print('image ' + str(item))

    elif FLAGS.mode == 'mul_eval':
        print('---------Start multi-scale eval-------------')
        crop_size_h = cfg.IMAGE_HEIGHT # 480 512 500 224
        crop_size_w = cfg.IMAGE_WIDTH
        print('crop size: ' + str(crop_size_h))
        stride = int(crop_size_w / 3)
        mean = [0.485, 0.456, 0.406]  # rgb
        std = [0.229, 0.224, 0.225]
        mean_bgr = [103.94, 116.78, 123.68]

        # crf = False
        # if crf:
        #     print('use CRF')
        #     image_mean = tf.constant(mean, dtype=tf.float32)
        #     image_std = tf.constant(std, dtype=tf.float32)
        #     image_origin = tf.cast((image * image_std + image_mean) * 255, tf.uint8)
        #     logits = tf.nn.softmax(logits)
        #     logits = tf.py_func(dense_crf_batch, [logits, image_origin], tf.float32)


        if not os.path.exists(cfg.save_dir + 'output'):
            os.mkdir(cfg.save_dir + 'output')
        f = open(cfg.save_dir + 'output/result.txt', 'w')

        total_acc_cls = []
        total_tp_num = []
        total_all_num = []

        total_tps = []
        total_fps = []
        total_fns = []

        hist = np.zeros((cfg.NUM_OF_CLASSESS, cfg.NUM_OF_CLASSESS))

        import cv2
        for item in range(len(img_list)):
            valid_images = [cv2.imread(img_list[item])]
            valid_annotations = [np.expand_dims(misc.imread(label_list[item]), axis=2)]
            im_name = img_list[item].split('/')[-1].split('.')[0]
            img_ori = valid_images[0]
            h_ori, w_ori, _ = img_ori.shape

            if 'ecp' in cfg.dataset:
                scs = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            elif 'RueMonge' in cfg.dataset:
                scs = [0.45, 0.6, 0.75]
                # scs = [0.75]

            # scs = [0.75, 1.0, 1.25, 1.5, 1.75]
            # scs = [1.25]

            maps = []
            for sc in scs:
                img = cv2.resize(img_ori, (int(float(w_ori) * sc), int(float(h_ori) * sc)), interpolation=cv2.INTER_LINEAR)
                score_map = data_crop_test_output(sess, image, logits, img, mean, std, mean_bgr, crop_size_h,
                                                  crop_size_w, stride)
                score_map = cv2.resize(score_map, (w_ori, h_ori), interpolation=cv2.INTER_LINEAR)
                maps.append(score_map)
            score_map = np.mean(np.stack(maps), axis=0)

            maps2 = []
            for sc in scs:
                img2 = cv2.resize(img_ori, (int(float(w_ori) * sc), int(float(h_ori) * sc)), interpolation=cv2.INTER_LINEAR)
                img2 = cv2.flip(img2, 1)
                # score_map2 = sess.run(logits, feed_dict={image: [img2], keep_probability: 1.0})[0]    # use for test full image
                score_map2 = data_crop_test_output(sess, image, logits, img2, mean, std, mean_bgr, crop_size_h,
                                                  crop_size_w, stride)
                score_map2 = cv2.resize(score_map2, (w_ori, h_ori), interpolation=cv2.INTER_LINEAR)
                maps2.append(score_map2)
            score_map2 = np.mean(np.stack(maps2), axis=0)
            score_map2 = cv2.flip(score_map2, 1)
            score_map = (score_map + score_map2) / 2

            # save score map for crf
            if False:
                if not os.path.exists(cfg.save_dir + '/score_map/'):
                    os.makedirs(cfg.save_dir + '/score_map/')
                np.save(cfg.save_dir + '/score_map/' + im_name + '.npy', score_map)

            pred_label = np.argmax(score_map, 2)


            pred_label = np.asarray(pred_label, dtype='uint8')
            pred = [pred_label[:, :, np.newaxis]]


            hist += fast_hist(valid_annotations[0].flatten(), pred[0].flatten(), cfg.NUM_OF_CLASSESS)  # gt, pred, class

            pred_vision(pred[0], im_name, cfg.dataset)
            pred_vision(valid_annotations[0], im_name + '_gt', cfg.dataset)
            print('image ' + str(item))
            f.write('image ' + im_name + '\n')
            f.write('scales: ' + str(scs) +'\n')
            for itr in range(FLAGS.batch_size):

                cls_acc, img_acc, tp_num, all_num = eval_img2(valid_annotations[itr], pred[itr])
                tps, fps, fns = eval_fscore(valid_annotations[itr], pred[itr])


                for cls in range(len(cls_acc)):
                    print(cfg.class_names[cls] + ': ' + str(cls_acc[cls]))
                    f.write(cfg.class_names[cls] + ': ' + str(cls_acc[cls]) + '\n')
                print('img-' + im_name+ ': ' + str(img_acc))
                f.write('img-' + im_name+ ' : ' + str(img_acc))
                print('-----------------------------')
                f.write('-------------------------------' + '\n')
                print('\n')
                f.write('\n')

            total_acc_cls.append(cls_acc)
            total_tp_num.append(tp_num)
            total_all_num.append(all_num)

            total_tps.append(tps)
            total_fps.append(fps)
            total_fns.append(fns)

        # overall accuracy  1
        # print('Shape hist: ', hist.shape)
        f.write('Shape hist: ' + str(hist.shape) + '\n')
        over_acc = np.diag(hist).sum() / hist.sum()
        print('1 overall accuracy', over_acc)
        f.write('1 overall accuracy' + str(over_acc) + '\n')

        # per-class accuracy
        acc = np.diag(hist) / hist.sum(1)
        print('1 mean accuracy', acc)
        f.write('1 mean accuracy' + str(acc) + '\n')

        # overall accuracy  2
        hist[0, :] = 0                                   # Ignore outlier
        # print('Shape hist: ', hist.shape)
        f.write('Shape hist: ' + str(hist.shape) + '\n')
        over_acc = np.diag(hist).sum() / hist.sum()
        print('2 overall accuracy', over_acc)
        f.write('2 overall accuracy' + str(over_acc) + '\n')

        # per-class accuracy
        acc = np.diag(hist) / hist.sum(1)
        print('2 mean accuracy', acc)
        f.write('2 mean accuracy' + str(acc) + '\n')

        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)
        # f1-score
        f1_scores = []
        for c in range(1, cfg.NUM_OF_CLASSESS):
            TP = hist[c][c]
            FP = np.sum(hist[:, c]) - hist[c][c]
            FN = np.sum(hist[c, :]) - hist[c][c]
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

            f1 = (2 * precision * recall) / (precision + recall)
            f1_scores.append(f1)

        mean_f1_score = sum(f1_scores) / len(f1_scores)
        print('f1_score: ' + str(mean_f1_score))
        f.write('f1 score: ' + str(mean_f1_score) + '\n')

        # per-class IU
        numerator = np.diag(hist)
        denominator = hist.sum(1) + hist.sum(0) - np.diag(hist)
        numerator_noBg = np.delete(numerator, 0, axis=0)
        denominator_noBg = np.delete(denominator, 0, axis=0)
        iu = numerator_noBg / denominator_noBg
        print('IoU ' + str(iu))
        f.write('IoU ' + str(iu) + '\n')
        print('mean IoU ', np.nanmean(iu))
        f.write('mean IoU ' + str(np.nanmean(iu)) + '\n')

        total_tps = np.array(total_tps)
        total_fps = np.array(total_fps)
        total_fns = np.array(total_fns)
        F1_socre2 = []
        for column in range(total_tps.shape[1]):

            cls_tp = []
            cls_fp = []
            cls_fn = []

            for row in range(total_tps.shape[0]):

                cls_tp.append(total_tps[row][column])
                cls_fp.append(total_fps[row][column])
                cls_fn.append(total_fns[row][column])
            prec = sum(cls_tp) / (sum(cls_tp) + sum(cls_fp))
            rec = sum(cls_tp) / (sum(cls_tp) + sum(cls_fn))
            # print(cfg.class_names[column] + '-prec:' + str(prec) + ', rec: ' + str(rec))
            F1_socre2.append((2 * prec * rec) / (prec + rec))
        # print('F1-score2: ' + str(sum(F1_socre2) / len(F1_socre2)))

        total_acc_cls = np.array(total_acc_cls)
        total_tp_num = np.array(total_tp_num)
        total_all_num = np.array(total_all_num)
        print('Total Accuracy: ')
        f.write('Total Accuracy: \n')

        filename = cfg.save_dir + 'output/acc.csv'
        f_csv = open(filename, 'w')
        writer = csv.writer(f_csv)

        class_avg_acc = []
        for column in range(total_acc_cls.shape[1]):

            cls_tp_num = []
            cls_all_num = []

            for row in range(total_acc_cls.shape[0]):

                cls_tp_num.append(total_tp_num[row][column])
                cls_all_num.append(total_all_num[row][column])

            class_acc = sum(cls_tp_num) / sum(cls_all_num)
            print(cfg.class_names[column] + '-acc:' + str(class_acc))
            f.write(cfg.class_names[column] + '-acc:' + str(class_acc) + '\n')
            writer.writerow([cfg.class_names[column], str(class_acc)])
            class_avg_acc.append(class_acc)


        print('\nTotal Acc:' + str(np.sum(total_tp_num) / np.sum(total_all_num)))
        f.write('\nTotal Acc:' + str(np.sum(total_tp_num) / np.sum(total_all_num)) + '\n')
        print('\nMean Acc:' + str(sum(class_avg_acc) / len(class_avg_acc)))
        f.write('\nMean Acc:' + str(sum(class_avg_acc) / len(class_avg_acc)) + '\n')

        writer.writerow(['Total acc', str(np.sum(total_tp_num) / np.sum(total_all_num))])
        writer.writerow(['Mean acc', str(sum(class_avg_acc) / len(class_avg_acc))])
        writer.writerow(['Mean_f1_score', str(mean_f1_score)])
        writer.writerow(['Mean IoU', str(np.nanmean(iu))])


        f_csv.close()
        f.close()


if __name__ == "__main__":
    tf.app.run()