# Config file for define some parameters
import argparse
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Facade ALK Network")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--image_height", type=int, default=512,
                        help="Image height and width of image.")
    parser.add_argument("--image_width", type=int, default=512,
                        help="Image height and width of image.")
    parser.add_argument("--learning_rate", type=float,
                        default=2e-4,
                        help="Learning rate for training.")
    parser.add_argument("--optimizer", type=str, default='Adam',    # Adam Momentum
                        help="optimizer for BP.")

    return parser.parse_args()
args = get_arguments()

# ---------------Modified Paras---------------------------
dataset = 'RueMonge2014' # 'cmp' # 'ecp_compare5_aug2'  #
use_gpu = '1'

NUM_OF_CLASSESS = 8                 # Modified 8 or 9 or 5 for Rue, ECP, CMP

Gradient_Accumulation = 1           # The number of gradient accumulation

total_iter = 10000

model_save_num = 12

is_epoch_acc = False

is_time_acc = False
# 10s to show acc
acc_interval = 180                  # 120s for ECP, 180 for RueMonge

# epoch 10000 to show train data acc
start_show_iter = 2000

is_save_epoch = False
save_epoch_inter = 100
start_save_epoch = 500

is_save_step = True
save_step_inter = 2000
start_save_step = 6000

weight_decay = 0.0001

freeze_bn = True

is_save_last10_model = False
# ----------------------------------------------------------


if dataset == 'ecp0':
    data_dir = '/media/ilab/Storage 2/PycharmProjects/Data_pre-processing-ECP/ecp/'
    save_dir = 'saveUntitled Folders/ecp/'
    logs_dir = '/home/ilab/tensorboard/ecp/'
    class_names = ['Outlier','Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
    train_number = 84
elif dataset == 'ecp':
    data_dir = '/media/ilab/Storage 2/PycharmProjects/Data_pre-processing-ECP/5_ecp_100/ecp_4/'
    save_dir = 'saves/0_test_coarse/'
    logs_dir = '/home/ilab/tensorboard/0_test_coarse/'
    class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
    train_number = 80

elif dataset == 'cmp':
    data_dir = 'data/cmp/'
    save_dir = 'saves/cmp/'
    logs_dir = '/home/ilab/tensorboard/cmp/'
    class_names = ['Outlier', 'Wall', 'Window', 'Door', 'Balcony']
    train_number = 606

elif dataset == 'RueMonge2014':     # [0.4, 1.0]
    data_dir = 'data/RueMonge2014/'               # 8
    save_dir = 'saves/RueMonge2014_pyramid_alknet/'
    logs_dir = 'tensorboard/RueMonge2014/'
    class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Sky', 'Shop']
    train_number = 113

elif dataset == 'camvid':
    data_dir = 'data/camvid/'
    save_dir = 'saves/camvid_psp/'
    logs_dir = 'tensorboard/camvid/'
    class_names = ['Outlier', 'Sky', 'Building', 'Pole', 'Road', 'Pavement',
                   'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist']
    train_number = 367

elif dataset == 'ecp_compare1_aug_sam':
    data_dir = 'data/ecp_compare/ecp_1_aug_sam/'
    save_dir = 'saves/ecp_compare1_aug_sam_pyalk/'                          # Modified
    logs_dir = '/home/ilab/tensorboard/ecp_compare1/'         # MOdified
    class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
    train_number = 4760

elif dataset == 'ecp_compare1_aug2':
    data_dir = 'data/ecp_compare/ecp_1_aug2/'
    save_dir = 'saves/ecp_compare1_aug2_danet/'                          # Modified
    logs_dir = 'tensorboard/ecp_compare1/'         # MOdified
    class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
    train_number = 3074
elif dataset == 'ecp_compare2_aug2':
    data_dir = 'data/ecp_compare/ecp_2_aug2/'
    save_dir = 'saves/ecp_compare2_aug2_danet/'                          # Modified
    logs_dir = 'tensorboard/ecp_compare2/'         # MOdified
    class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
    train_number = 3122
elif dataset == 'ecp_compare3_aug2':
    data_dir = 'data/ecp_compare/ecp_3_aug2/'
    save_dir = 'saves/ecp_compare3_aug2_danet/'                          # Modified
    logs_dir = 'tensorboard/ecp_compare3/'         # MOdified
    class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
    train_number = 3150
elif dataset == 'ecp_compare4_aug2':
    data_dir = 'data/ecp_compare/ecp_4_aug2/'
    save_dir = 'saves/ecp_compare4_aug2_danet/'                          # Modified
    logs_dir = 'tensorboard/ecp_compare4/'         # MOdified
    class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
    train_number = 3012
elif dataset == 'ecp_compare5_aug2':
    data_dir = 'data/ecp_compare/ecp_5_aug2/'
    save_dir = 'saves/ecp_compare5_aug2_deeplabv3_plus/'                          # Modified
    logs_dir = 'tensorboard/ecp_compare5/'         # MOdified
    class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
    train_number = 3118

# elif dataset == 'ecp_compare1_aug':
#     data_dir = 'data/ecp_compare/ecp_1_aug/'
#     save_dir = 'saves/ecp_compare1_aug_Pyramid_ALKNet/'                          # Modified
#     logs_dir = 'tensorboard/ecp_compare1/'         # MOdified
#     class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
#     train_number = 3084
# elif dataset == 'ecp_compare2_aug':
#     data_dir = 'data/ecp_compare/ecp_2_aug/'
#     save_dir = 'saves/ecp_compare2_aug_Pyramid_ALKNet/'                          # Modified
#     logs_dir = 'tensorboard/ecp_compare2/'         # MOdified
#     class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
#     train_number = 3098
# elif dataset == 'ecp_compare3_aug':
#     data_dir = 'data/ecp_compare/ecp_3_aug/'
#     save_dir = 'saves/ecp_compare3_aug_Pyramid_ALKNet/'                          # Modified
#     logs_dir = 'tensorboard/ecp_compare3/'         # MOdified
#     class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
#     train_number = 3136
# elif dataset == 'ecp_compare4_aug':
#     data_dir = 'data/ecp_compare/ecp_4_aug/'
#     save_dir = 'saves/ecp_compare4_aug_Pyramid_ALKNet/'                          # Modified
#     logs_dir = 'tensorboard/ecp_compare4/'         # MOdified
#     class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
#     train_number = 3012
# elif dataset == 'ecp_compare5_aug':
#     data_dir = 'data/ecp_compare/ecp_5_aug/'
#     save_dir = 'saves/ecp_compare5_aug_Pyramid_ALKNet/'                          # Modified
#     logs_dir = 'tensorboard/ecp_compare5/'         # MOdified
#     class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
#     train_number = 3130

elif dataset == 'ecp_compare1_com':             # Acc no increase
    data_dir = 'data_augmentation/ecp_1/'
    save_dir = 'saves/ecp_compare1_com_res50/'                          # Modified
    logs_dir = '/home/ilab/tensorboard/ecp_compare1_com/'         # MOdified
    class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
    train_number = 504
# data augmentation 1 is not use in experiments, 2 is used
elif dataset == 'ecp_compare1':
    data_dir = 'data/ecp_compare/ecp_1/'
    save_dir = 'saves/ecp_compare1_pyalk_multiloss/'                          # Modified
    logs_dir = '/home/ilab/tensorboard/ecp_compare1/'         # MOdified
    class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
    train_number = 84
elif dataset == 'ecp_compare2':
    data_dir = 'data/ecp_compare/ecp_2/'
    save_dir = 'saves/ecp_compare2_pyalk/'                          # Modified
    logs_dir = '/home/ilab/tensorboard/ecp_compare2/'         # MOdified
    class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
    train_number = 84
elif dataset == 'ecp_compare3':
    data_dir = 'data/ecp_compare/ecp_3/'
    save_dir = 'saves/ecp_compare3_pyalk_multiloss_ds/'                          # Modified
    logs_dir = '/home/ilab/tensorboard/ecp_compare3/'         # MOdified
    class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
    train_number = 84
elif dataset == 'ecp_compare4':
    data_dir = 'data/ecp_compare/ecp_4/'
    save_dir = 'saves/ecp_compare4_pyalk/'                          # Modified
    logs_dir = '/home/ilab/tensorboard/ecp_compare4/'         # MOdified
    class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
    train_number = 84
elif dataset == 'ecp_compare5':
    data_dir = 'data/ecp_compare/ecp_5/'
    save_dir = 'saves/ecp_compare5_pyalk/'                          # Modified
    logs_dir = '/home/ilab/tensorboard/ecp_compare5/'         # MOdified
    class_names = ['Outlier', 'Window', 'Wall', 'Balcony', 'Door', 'Roof', 'Chimney', 'Sky', 'Shop']
    train_number = 84


# --------fixed------------

# pre_trained_model = '/media/ilab/Storage 2/PycharmProjects/tensorflow-deeplab-v3-plus/model/model.ckpt-63505'
pre_trained_model = '/media/ilab/Storage 2/mwg_tf_pretrained/resnet_v1_50.ckpt'

IMAGE_HEIGHT = args.image_height
IMAGE_WIDTH = args.image_width

batch_size = args.batch_size

learning_rate = args.learning_rate

optimizer = args.optimizer

decay_rate = 0.9

summary_interval = 5  # 60s to save a summary

train_data_dir = data_dir + 'train'
train_data_list = data_dir + 'train.txt'

val_data_dir = data_dir + 'train' #'train80'
val_data_list = data_dir + 'train.txt' #'train80.txt'

test_data_dir = data_dir + 'val' # 'test_imgs'#
test_data_list = data_dir + 'val.txt' # 'test_imgs/test2.txt'#

random_resize = False
random_color = False

random_scale = True      # False for ECP, True for Rue
minScale = 0.4            # Scales of Rue
maxScale = 1.1            # Scales of Rue
random_mirror = True     # False for ECP, True for Rue
random_crop_pad = True   # False for ECP, True for Rue
ignore_label = 0


# -------------------------


# is use a epoch to run
# ---------------------------------------------------------------
train_file = open(train_data_list)
train_list = []
for line in train_file:
    train_list.append(line)

circle = int(total_iter * batch_size * Gradient_Accumulation / len(train_list)) + 1
new_train_file = open(data_dir + 'new_train.txt', 'w')
import random
for i in range(circle):
    random.shuffle(train_list)
    for txt in train_list:
        new_train_file.write(txt)
print('Generate txt OK, use new train file！')

train_file.close()
new_train_file.close()
train_data_list = data_dir + 'new_train.txt'
# ----------------------------------------------------------------


import numpy as np

IMG_MEAN = np.array([103.94, 116.78, 123.68], dtype=np.float32)     # B G R


import tensorflow as tf
def get_cur_lr(step_ph):
    cur_lr = tf.py_func(_get_cur_lr, [step_ph], tf.float32)
    return cur_lr
def _get_cur_lr(step_ph):
    step = np.array(step_ph, np.int32)
    ep = int(step / (train_number / batch_size))
    if ep < 10:
        cur_lr = 1e-4
    elif ep < 20:
        cur_lr = 1e-5
    else:
        cur_lr = 1e-6

    return np.asarray(cur_lr, dtype=np.float32)

def get_step_lr(step_ph):
    step_lr = tf.py_func(_get_step_lr, [step_ph], tf.float32)
    return step_lr
def _get_step_lr(step_ph):
    step = np.array(step_ph, np.int32)
    ep = step
    if ep < 500:
        step_lr = 2e-4
    elif ep < 1000:
        step_lr = 1e-4
    elif ep < 1500:
        step_lr = 5e-5
    elif ep < 2000:
        step_lr = 2.5e-5
    elif ep < 2500:
        step_lr = 1e-5
    elif ep < 3000:
        step_lr = 5e-6
    elif ep < 3500:
        step_lr = 2.5e-6
    elif ep < 4000:
        step_lr = 1e-6
    elif ep < 4500:
        step_lr = 5e-7
    else:
        step_lr = 2.5e-7

    return np.asarray(step_lr, dtype=np.float32)

def get_cosine_lr(step_ph):
    cur_lr = tf.py_func(_get_cosine_lr, [step_ph], tf.float32)
    return cur_lr
import math
def _get_cosine_lr(step_ph):
    step = np.array(step_ph, np.int32)
    total_step = int((train_number / batch_size) * args.epoch_num)
    cur_lr = ((1 + math.cos((step * 3.1415926535897932384626433) / total_step)) * args.learning_rate) / 2
    return np.asarray(cur_lr, dtype=np.float32)

def noam_scheme(cur_step):                                  # warmup learning rate
    lr = tf.py_func(_noam_scheme, [cur_step], tf.float32)
    return lr

def _noam_scheme(cur_step):
    """
    if cur < warnup_step, lr increase
    if cur > warnup_step, lr decrease
    """
    step = np.array(cur_step, np.int32)
    init_lr = learning_rate
    global_step = total_iter
    warnup_factor = 1.0 / 3
    power = 0.9
    warnup_step = 500

    if step <= warnup_step:
        alpha = step / warnup_step
        warnup_factor = warnup_factor * (1 - alpha) + alpha
        lr = init_lr * warnup_factor

    else:
        # learning_rate = tf.scalar_mul(init_lr, tf.pow((1 - cur_step / global_step), power))
        lr = init_lr * np.power(
            (1 - (step - warnup_step) / (global_step - warnup_step)), power)

    return np.asarray(lr, dtype=np.float32)

def circle_scheme(cur_step):                                # circle learning rate
    lr = tf.py_func(_circle_scheme, [cur_step], tf.float32)
    return lr

def _circle_scheme(cur_step):
    step = np.array(cur_step, np.int32)
    CYCLE = 1000
    LR_INIT = learning_rate
    LR_MIN = 1e-10
    scheduler = lambda x: ((LR_INIT - LR_MIN) / 2) * (np.cos(3.1415926535897932384626433 * (np.mod(x - 1, CYCLE) / (CYCLE))) + 1) + LR_MIN
    lr = scheduler(step)

    return np.asarray(lr, dtype=np.float32)