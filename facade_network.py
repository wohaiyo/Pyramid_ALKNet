from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim
import config as cfg
from tensorflow.python.ops import nn
import math
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
from tensorflow.contrib.slim.nets import resnet_v2

def sum_features(conv_output):
    '''sum feature value'''
    num_or_size_splits = conv_output.get_shape().as_list()[-1]
    each_convs = tf.split(conv_output, num_or_size_splits, axis=3)
    first_convs = each_convs[0]
    for i in range(1, num_or_size_splits):
        first_convs = tf.add(first_convs, each_convs[i])
    return first_convs

def resudual_block(x, name=None):
    x_shape = tf.shape(x)
    channel = x.get_shape().as_list()[-1]
    res = slim.conv2d(x, channel, [3, 3], scope=name + '_1', padding='SAME')
    res = slim.conv2d(res, channel, [3, 3], scope=name + '_2', padding='SAME', activation_fn=None)
    return tf.nn.relu(tf.add(x, res))

def resudual_block_channel(x, channel, name=None):
    res = slim.conv2d(x, channel, [3, 3], scope=name + '_1', padding='SAME')
    res = slim.conv2d(res, channel, [3, 3], scope=name + '_2', padding='SAME', activation_fn=None)
    x = slim.conv2d(x, channel, [1, 1], scope=name + '_down', padding='SAME')
    return tf.nn.relu(tf.add(x, res))

def channel_attention0(low, high, low_channel, name=None):
    con = tf.concat([high, low], axis=3)
    pool_size = tf.shape(low)
    global_context = slim.avg_pool2d(con, [pool_size[1], pool_size[2]])
    weight = slim.conv2d(global_context, low_channel, [1, 1])
    weight = slim.conv2d(weight, low_channel, [1, 1],activation_fn=nn.sigmoid)
    return weight * low

def conv2d_trans(x, W, b, output_shape, stride=2):
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def weight_variable(shape, stddev=0.02, name=None, is_training=True):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial, trainable=is_training)
    else:
        return tf.get_variable(name, initializer=initial, trainable=is_training)

def bias_variable(shape, name=None, is_training=True):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial, trainable=is_training)
    else:
        return tf.get_variable(name, initializer=initial, trainable=is_training)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)

def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))

def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def conv2d_transpose_strided_Nobaises(x, W, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return conv

# Spatial LSTM
def BiDirect_LSTM(x, state_size=256):
    '''x: input feature'''
    # add message passing #
    # top down

    # query = tf.transpose(query, [0, 3, 1, 2])
    # query = tf.reshape(query, [batch_size, key_channels, -1])

    batch_size, h, w, c = x.get_shape().as_list()

    h_list = []
    with tf.variable_scope('layer_lstm_refine_top_down'):
        # Bidirectional LSTM
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(state_size)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(state_size)
        inputf = tf.transpose(x, [0, 2, 1, 3])
        inputf = tf.reshape(inputf, [batch_size*w, h, c])

        outputs, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputf, dtype=inputf.dtype)
        output = tf.concat([outputs[0], outputs[1]], axis=2)
    x = tf.reshape(output, [batch_size, w, h, c])   # b×w, h, c
    x = tf.transpose(x, [0, 2, 1, 3])               # b, h , w, c


    with tf.variable_scope('layer_lstm_refine_left_right'):
        # Bidirectional LSTM
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(state_size)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(state_size)
        inputf = tf.reshape(inputf, [batch_size * h, w, c])

        outputs, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputf, dtype=inputf.dtype)
        output = tf.concat([outputs[0], outputs[1]], axis=2)
    out = tf.reshape(output, [batch_size, h, w, c])  # b, h, w, c

    return out

def ReNet(x, C = 128):
    k = 1
    feature_list_old = []
    feature_list_new = []
    # top to down
    for cnt in range(x.get_shape().as_list()[1]):
        feature_list_old.append(tf.expand_dims(x[:, cnt, :, :], axis=1))
    feature_list_new.append(tf.expand_dims(x[:, 0, :, :], axis=1))

    w1 = tf.get_variable('W1', [1, 1, C, C],
                         initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (1 * C * C * 5))))
    with tf.variable_scope("convs_6_1"):
        conv_6_1 = tf.nn.relu(tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w1, [1, 1, 1, 1], 'SAME')),
                          feature_list_old[1]))
        feature_list_new.append(conv_6_1)

    for cnt in range(2, x.get_shape().as_list()[1]):
        with tf.variable_scope("convs_6_1", reuse=True):
            conv_6_1 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w1, [1, 1, 1, 1], 'SAME')),
                              feature_list_old[cnt])
            feature_list_new.append(conv_6_1)
    top_down = tf.stack(feature_list_new, axis=1)
    top_down = tf.squeeze(top_down, axis=2)

    # down to top #
    feature_list_old = []
    for cnt in range(x.get_shape().as_list()[1]):
        feature_list_old.append(tf.expand_dims(x[:, cnt, :, :], axis=1))
    feature_list_new = []
    length = int(cfg.IMAGE_HEIGHT / 8) - 1
    feature_list_new.append(feature_list_old[length])

    w2 = tf.get_variable('W2', [1, k, C, C],
                         initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (k * C * C * 5))))
    with tf.variable_scope("convs_6_2"):
        conv_6_2 = tf.nn.relu(tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w2, [1, 1, 1, 1], 'SAME')),
                          feature_list_old[length - 1]))
        feature_list_new.append(conv_6_2)

    for cnt in range(2, x.get_shape().as_list()[1]):
        with tf.variable_scope("convs_6_2", reuse=True):
            conv_6_2 = tf.nn.relu(tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w2, [1, 1, 1, 1], 'SAME')),
                              feature_list_old[length - cnt]))
            feature_list_new.append(conv_6_2)

    feature_list_new.reverse()

    down_top = tf.stack(feature_list_new, axis=1)
    down_top = tf.squeeze(down_top, axis=2)

    # left to right #

    feature_list_old = []
    feature_list_new = []
    for cnt in range(x.get_shape().as_list()[2]):
        feature_list_old.append(tf.expand_dims(x[:, :, cnt, :], axis=2))
    feature_list_new.append(tf.expand_dims(x[:, :, 0, :], axis=2))

    w3 = tf.get_variable('W3', [k, 1, C, C],
                         initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (k * C * C * 5))))
    with tf.variable_scope("convs_6_3"):
        conv_6_3 = tf.nn.relu(tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w3, [1, 1, 1, 1], 'SAME')),
                          feature_list_old[1]))
        feature_list_new.append(conv_6_3)

    for cnt in range(2, x.get_shape().as_list()[2]):
        with tf.variable_scope("convs_6_3", reuse=True):
            conv_6_3 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w3, [1, 1, 1, 1], 'SAME')),
                              feature_list_old[cnt])
            feature_list_new.append(conv_6_3)

    left_right = tf.stack(feature_list_new, axis=2)
    left_right = tf.squeeze(left_right, axis=3)

    # right to left #

    feature_list_old = []
    for cnt in range(x.get_shape().as_list()[2]):
        feature_list_old.append(tf.expand_dims(x[:, :, cnt, :], axis=2))
    feature_list_new = []
    length = int(cfg.IMAGE_WIDTH / 8) - 1
    feature_list_new.append(feature_list_old[length])

    w4 = tf.get_variable('W4', [k, 1, C, C],
                         initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (k * C * C * 5))))
    with tf.variable_scope("convs_6_4"):
        conv_6_4 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w4, [1, 1, 1, 1], 'SAME')),
                          feature_list_old[length - 1])
        feature_list_new.append(conv_6_4)

    for cnt in range(2, x.get_shape().as_list()[2]):
        with tf.variable_scope("convs_6_4", reuse=True):
            conv_6_4 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w4, [1, 1, 1, 1], 'SAME')),
                              feature_list_old[length - cnt])
            feature_list_new.append(conv_6_4)

    feature_list_new.reverse()
    right_left = tf.stack(feature_list_new, axis=2)
    right_left = tf.squeeze(right_left, axis=3)

    feature_concat = tf.concat([top_down, down_top, left_right, right_left], axis=3)
    processed_feature = slim.conv2d(feature_concat, C, [1, 1], scope='direction_down_dim')

    return processed_feature

# Spatial CNN
def Spatial_CNN(x, C=128):
    # added part of SCNN #

    # conv stage 5_4
    conv_5_4 = slim.conv2d(x, 1024, [3, 3], scope='conv5_4', rate=12)    # 12 for seg

    # conv stage 5_5
    conv_5_5 = slim.conv2d(conv_5_4, C, [1, 1], scope='conv5_5')  # 8 x 36 x 100 x 128

    # add message passing #

    # top to down #

    feature_list_old = []
    feature_list_new = []
    for cnt in range(conv_5_5.get_shape().as_list()[1]):
        feature_list_old.append(tf.expand_dims(conv_5_5[:, cnt, :, :], axis=1))
    feature_list_new.append(tf.expand_dims(conv_5_5[:, 0, :, :], axis=1))

    w1 = tf.get_variable('W1', [1, 9, 128, 128],
                         initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
    with tf.variable_scope("convs_6_1"):
        conv_6_1 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w1, [1, 1, 1, 1], 'SAME')),
                          feature_list_old[1])
        feature_list_new.append(conv_6_1)

    for cnt in range(2, conv_5_5.get_shape().as_list()[1]):
        with tf.variable_scope("convs_6_1", reuse=True):
            conv_6_1 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w1, [1, 1, 1, 1], 'SAME')),
                              feature_list_old[cnt])
            feature_list_new.append(conv_6_1)

    # down to top #
    feature_list_old = feature_list_new
    feature_list_new = []
    length = int(cfg.IMAGE_HEIGHT / 8) - 1
    feature_list_new.append(feature_list_old[length])

    w2 = tf.get_variable('W2', [1, 9, 128, 128],
                         initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
    with tf.variable_scope("convs_6_2"):
        conv_6_2 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w2, [1, 1, 1, 1], 'SAME')),
                          feature_list_old[length - 1])
        feature_list_new.append(conv_6_2)

    for cnt in range(2, conv_5_5.get_shape().as_list()[1]):
        with tf.variable_scope("convs_6_2", reuse=True):
            conv_6_2 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w2, [1, 1, 1, 1], 'SAME')),
                              feature_list_old[length - cnt])
            feature_list_new.append(conv_6_2)

    feature_list_new.reverse()

    processed_feature = tf.stack(feature_list_new, axis=1)
    processed_feature = tf.squeeze(processed_feature, axis=2)

    # left to right #

    feature_list_old = []
    feature_list_new = []
    for cnt in range(processed_feature.get_shape().as_list()[2]):
        feature_list_old.append(tf.expand_dims(processed_feature[:, :, cnt, :], axis=2))
    feature_list_new.append(tf.expand_dims(processed_feature[:, :, 0, :], axis=2))

    w3 = tf.get_variable('W3', [9, 1, 128, 128],
                         initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
    with tf.variable_scope("convs_6_3"):
        conv_6_3 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w3, [1, 1, 1, 1], 'SAME')),
                          feature_list_old[1])
        feature_list_new.append(conv_6_3)

    for cnt in range(2, processed_feature.get_shape().as_list()[2]):
        with tf.variable_scope("convs_6_3", reuse=True):
            conv_6_3 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w3, [1, 1, 1, 1], 'SAME')),
                              feature_list_old[cnt])
            feature_list_new.append(conv_6_3)

    # right to left #

    feature_list_old = feature_list_new
    feature_list_new = []
    length = int(cfg.IMAGE_WIDTH / 8) - 1
    feature_list_new.append(feature_list_old[length])

    w4 = tf.get_variable('W4', [9, 1, 128, 128],
                         initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
    with tf.variable_scope("convs_6_4"):
        conv_6_4 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w4, [1, 1, 1, 1], 'SAME')),
                          feature_list_old[length - 1])
        feature_list_new.append(conv_6_4)

    for cnt in range(2, processed_feature.get_shape().as_list()[2]):
        with tf.variable_scope("convs_6_4", reuse=True):
            conv_6_4 = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w4, [1, 1, 1, 1], 'SAME')),
                              feature_list_old[length - cnt])
            feature_list_new.append(conv_6_4)

    feature_list_new.reverse()
    processed_feature = tf.stack(feature_list_new, axis=2)
    processed_feature = tf.squeeze(processed_feature, axis=3)


    return processed_feature

def dilated_conv(input_tensor, k_size, out_dims, name, dilation=1, pad='SAME'):
        """
        将卷积和激活封装在一起
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :return:
        """
        with tf.variable_scope(name):
            in_shape = input_tensor.get_shape().as_list()
            in_channel = in_shape[3]
            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"

            padding = pad.upper()

            if isinstance(k_size, list):
                filter_shape = [k_size[0], k_size[1]] + [in_channel, out_dims]
            else:
                filter_shape = [k_size, k_size] + [in_channel, out_dims]

            w_init = initializers.xavier_initializer()
            b_init = tf.constant_initializer()

            w = tf.get_variable('weights', filter_shape, initializer=w_init)
            b = tf.get_variable('biases', [out_dims], initializer=b_init)

            conv = tf.nn.atrous_conv2d(value=input_tensor, filters=w, rate=dilation,
                                       padding=padding, name='dilation_conv')
            conv = tf.add(conv, b)
        relu = tf.nn.relu(conv, name='/relu')

        return relu

def spatial_cnn2(input_feature, k, c):   # 4x64x48x128
    '''

    :param self:
    :param input_feature:
    :param k: conv kernel size
    :param c:  input tensor channel
    :return:
    '''
    # add message passing #
    # top to down #

    feature_list_old = []
    feature_list_new = []
    for cnt in range(input_feature.get_shape().as_list()[1]):
        feature_list_old.append(tf.expand_dims(input_feature[:, cnt, :, :], axis=1))
    feature_list_new.append(tf.expand_dims(input_feature[:, 0, :, :], axis=1))

    w1 = tf.get_variable('W1', [1, k, c, c],
                         initializer=initializers.xavier_initializer())
    with tf.variable_scope("convs_top_down"):
        conv_top_down = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w1, [1, 1, 1, 1], 'SAME')),
                          feature_list_old[1])
        feature_list_new.append(conv_top_down)

    for cnt in range(2, input_feature.get_shape().as_list()[1]):
        with tf.variable_scope("convs_top_down", reuse=True):
            conv_top_down = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w1, [1, 1, 1, 1], 'SAME')),
                              feature_list_old[cnt])
            feature_list_new.append(conv_top_down)

    # down to top #
    feature_list_old = feature_list_new
    feature_list_new = []
    length = int(cfg.IMAGE_HEIGHT / 8) - 1
    feature_list_new.append(feature_list_old[length])

    w2 = tf.get_variable('W2', [1, k, c, c],
                         initializer=initializers.xavier_initializer())
    with tf.variable_scope("convs_down_top"):
        conv_down_top = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w2, [1, 1, 1, 1], 'SAME')),
                          feature_list_old[length - 1])
        feature_list_new.append(conv_down_top)

    for cnt in range(2, input_feature.get_shape().as_list()[1]):
        with tf.variable_scope("convs_down_top", reuse=True):
            conv_down_top = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w2, [1, 1, 1, 1], 'SAME')),
                              feature_list_old[length - cnt])
            feature_list_new.append(conv_down_top)

    feature_list_new.reverse()

    processed_feature = tf.stack(feature_list_new, axis=1)
    processed_feature = tf.squeeze(processed_feature, axis=2)

    # left to right #
    feature_list_old = []
    feature_list_new = []
    for cnt in range(processed_feature.get_shape().as_list()[2]):
        feature_list_old.append(tf.expand_dims(processed_feature[:, :, cnt, :], axis=2))
    feature_list_new.append(tf.expand_dims(processed_feature[:, :, 0, :], axis=2))

    w3 = tf.get_variable('W3', [k, 1, c, c],
                         initializer=initializers.xavier_initializer())
    with tf.variable_scope("convs_left_right"):
        conv_left_right = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w3, [1, 1, 1, 1], 'SAME')),
                          feature_list_old[1])
        feature_list_new.append(conv_left_right)

    for cnt in range(2, processed_feature.get_shape().as_list()[2]):
        with tf.variable_scope("convs_left_right", reuse=True):
            conv_left_right = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w3, [1, 1, 1, 1], 'SAME')),
                              feature_list_old[cnt])
            feature_list_new.append(conv_left_right)

    # right to left #

    feature_list_old = feature_list_new
    feature_list_new = []
    length = int(cfg.IMAGE_WIDTH / 8) - 1
    feature_list_new.append(feature_list_old[length])

    w4 = tf.get_variable('W4', [k, 1, c, c],
                         initializer=initializers.xavier_initializer())
    with tf.variable_scope("convs_right_left"):
        conv_right_left = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w4, [1, 1, 1, 1], 'SAME')),
                          feature_list_old[length - 1])
        feature_list_new.append(conv_right_left)

    for cnt in range(2, processed_feature.get_shape().as_list()[2]):
        with tf.variable_scope("convs_right_left", reuse=True):
            conv_right_left = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w4, [1, 1, 1, 1], 'SAME')),
                              feature_list_old[length - cnt])
            feature_list_new.append(conv_right_left)

    feature_list_new.reverse()
    processed_feature = tf.stack(feature_list_new, axis=2)
    processed_feature = tf.squeeze(processed_feature, axis=3)

    return processed_feature

def rnn_parsing_continue(input_feature, k, c):  # 4x64x48x128
    '''

    :param self:
    :param input_feature:
    :param k: conv kernel size
    :param c:  input tensor channel
    :return:
    '''
    # add message passing  feature continue#
    # top to down #

    feature_list_old = []
    feature_list_new = []
    for cnt in range(input_feature.get_shape().as_list()[1]):
        feature_list_old.append(tf.expand_dims(input_feature[:, cnt, :, :], axis=1))
    feature_list_new.append(tf.expand_dims(input_feature[:, 0, :, :], axis=1))

    w1 = tf.get_variable('W1', [1, k, c, c],
                         initializer=initializers.xavier_initializer())
    with tf.variable_scope("convs_top_down"):
        conv_top_down = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w1, [1, 1, 1, 1], 'SAME')),
                               feature_list_old[1])
        conv_top_down = tf.maximum(conv_top_down, 0)
        feature_list_new.append(conv_top_down)

    for cnt in range(2, input_feature.get_shape().as_list()[1]):
        with tf.variable_scope("convs_top_down", reuse=True):
            conv_top_down = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w1, [1, 1, 1, 1], 'SAME')),
                                   feature_list_old[cnt])
            conv_top_down = tf.maximum(conv_top_down, 0)
            feature_list_new.append(conv_top_down)

    # down to top #
    feature_list_old = feature_list_new
    feature_list_new = []
    length = int(cfg.IMAGE_HEIGHT / 8) - 1
    feature_list_new.append(feature_list_old[length])

    w2 = tf.get_variable('W2', [1, k, c, c],
                         initializer=initializers.xavier_initializer())
    with tf.variable_scope("convs_down_top"):
        conv_down_top = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w2, [1, 1, 1, 1], 'SAME')),
                               feature_list_old[length - 1])
        conv_down_top = tf.maximum(conv_down_top, 0)
        feature_list_new.append(conv_down_top)

    for cnt in range(2, input_feature.get_shape().as_list()[1]):
        with tf.variable_scope("convs_down_top", reuse=True):
            conv_down_top = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w2, [1, 1, 1, 1], 'SAME')),
                                   feature_list_old[length - cnt])
            conv_down_top = tf.maximum(conv_down_top, 0)
            feature_list_new.append(conv_down_top)

    feature_list_new.reverse()

    processed_feature = tf.stack(feature_list_new, axis=1)
    processed_feature = tf.squeeze(processed_feature, axis=2)

    # left to right #
    feature_list_old = []
    feature_list_new = []
    for cnt in range(processed_feature.get_shape().as_list()[2]):
        feature_list_old.append(tf.expand_dims(processed_feature[:, :, cnt, :], axis=2))
    feature_list_new.append(tf.expand_dims(processed_feature[:, :, 0, :], axis=2))

    w3 = tf.get_variable('W3', [k, 1, c, c],
                         initializer=initializers.xavier_initializer())
    with tf.variable_scope("convs_left_right"):
        conv_left_right = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w3, [1, 1, 1, 1], 'SAME')),
                                 feature_list_old[1])
        conv_left_right = tf.maximum(conv_left_right, 0)
        feature_list_new.append(conv_left_right)

    for cnt in range(2, processed_feature.get_shape().as_list()[2]):
        with tf.variable_scope("convs_left_right", reuse=True):
            conv_left_right = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w3, [1, 1, 1, 1], 'SAME')),
                                     feature_list_old[cnt])
            conv_left_right = tf.maximum(conv_left_right, 0)
            feature_list_new.append(conv_left_right)

    # right to left #

    feature_list_old = feature_list_new
    feature_list_new = []
    length = int(cfg.IMAGE_WIDTH / 8) - 1
    feature_list_new.append(feature_list_old[length])

    w4 = tf.get_variable('W4', [k, 1, c, c],
                         initializer=initializers.xavier_initializer())
    with tf.variable_scope("convs_right_left"):
        conv_right_left = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w4, [1, 1, 1, 1], 'SAME')),
                                 feature_list_old[length - 1])
        conv_right_left = tf.maximum(conv_right_left, 0)
        feature_list_new.append(conv_right_left)

    for cnt in range(2, processed_feature.get_shape().as_list()[2]):
        with tf.variable_scope("convs_right_left", reuse=True):
            conv_right_left = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w4, [1, 1, 1, 1], 'SAME')),
                                     feature_list_old[length - cnt])
            conv_right_left = tf.maximum(conv_right_left, 0)
            feature_list_new.append(conv_right_left)

    feature_list_new.reverse()
    processed_feature = tf.stack(feature_list_new, axis=2)
    processed_feature = tf.squeeze(processed_feature, axis=3)

    return processed_feature

def rnn_parsing_split(input_feature, k, c):  # 4x64x48x128
    '''

    :param self:
    :param input_feature:
    :param k: conv kernel size
    :param c:  input tensor channel
    :return:
    '''
    # add message passing  feature continue#
    # top to down #

    feature_list_old = []  # original feature

    feature_list_new = []
    for cnt in range(input_feature.get_shape().as_list()[1]):
        feature_list_old.append(tf.expand_dims(input_feature[:, cnt, :, :], axis=1))
    feature_list_new.append(tf.expand_dims(input_feature[:, 0, :, :], axis=1))

    w1 = tf.get_variable('W1', [1, k, c, c],
                         initializer=initializers.xavier_initializer())
    with tf.variable_scope("convs_top_down"):
        conv_top_down = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w1, [1, 1, 1, 1], 'SAME')),
                               feature_list_old[1])
        conv_top_down = tf.maximum(conv_top_down, 0)
        feature_list_new.append(conv_top_down)

    for cnt in range(2, input_feature.get_shape().as_list()[1]):
        with tf.variable_scope("convs_top_down", reuse=True):
            conv_top_down = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w1, [1, 1, 1, 1], 'SAME')),
                                   feature_list_old[cnt])
            conv_top_down = tf.maximum(conv_top_down, 0)
            feature_list_new.append(conv_top_down)

    feature_top_down = tf.stack(feature_list_new, axis=1)
    feature_top_down = tf.squeeze(feature_top_down, axis=2)

    # down to top #
    feature_list_new = []
    length = int(cfg.IMAGE_HEIGHT / 8) - 1
    feature_list_new.append(feature_list_old[length])

    w2 = tf.get_variable('W2', [1, k, c, c],
                         initializer=initializers.xavier_initializer())
    with tf.variable_scope("convs_down_top"):
        conv_down_top = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w2, [1, 1, 1, 1], 'SAME')),
                               feature_list_old[length - 1])
        conv_down_top = tf.maximum(conv_down_top, 0)
        feature_list_new.append(conv_down_top)

    for cnt in range(2, input_feature.get_shape().as_list()[1]):
        with tf.variable_scope("convs_down_top", reuse=True):
            conv_down_top = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w2, [1, 1, 1, 1], 'SAME')),
                                   feature_list_old[length - cnt])
            conv_down_top = tf.maximum(conv_down_top, 0)
            feature_list_new.append(conv_down_top)

    feature_list_new.reverse()

    feature_down_top = tf.stack(feature_list_new, axis=1)
    feature_down_top = tf.squeeze(feature_down_top, axis=2)

    # left to right #
    feature_list_old = []
    feature_list_new = []
    for cnt in range(input_feature.get_shape().as_list()[2]):
        feature_list_old.append(tf.expand_dims(input_feature[:, :, cnt, :], axis=2))
    feature_list_new.append(tf.expand_dims(input_feature[:, :, 0, :], axis=2))

    w3 = tf.get_variable('W3', [k, 1, c, c],
                         initializer=initializers.xavier_initializer())
    with tf.variable_scope("convs_left_right"):
        conv_left_right = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[0], w3, [1, 1, 1, 1], 'SAME')),
                                 feature_list_old[1])
        conv_left_right = tf.maximum(conv_left_right, 0)
        feature_list_new.append(conv_left_right)

    for cnt in range(2, input_feature.get_shape().as_list()[2]):
        with tf.variable_scope("convs_left_right", reuse=True):
            conv_left_right = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w3, [1, 1, 1, 1], 'SAME')),
                                     feature_list_old[cnt])
            conv_left_right = tf.maximum(conv_left_right, 0)
            feature_list_new.append(conv_left_right)
    feature_left_right = tf.stack(feature_list_new, axis=2)
    feature_left_right = tf.squeeze(feature_left_right, axis=3)

    # right to left #

    feature_list_new = []
    length = int(cfg.IMAGE_WIDTH / 8) - 1
    feature_list_new.append(feature_list_old[length])

    w4 = tf.get_variable('W4', [k, 1, c, c],
                         initializer=initializers.xavier_initializer())
    with tf.variable_scope("convs_right_left"):
        conv_right_left = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_old[length], w4, [1, 1, 1, 1], 'SAME')),
                                 feature_list_old[length - 1])
        conv_right_left = tf.maximum(conv_right_left, 0)
        feature_list_new.append(conv_right_left)

    for cnt in range(2, input_feature.get_shape().as_list()[2]):
        with tf.variable_scope("convs_right_left", reuse=True):
            conv_right_left = tf.add(tf.nn.relu(tf.nn.conv2d(feature_list_new[cnt - 1], w4, [1, 1, 1, 1], 'SAME')),
                                     feature_list_old[length - cnt])
            conv_right_left = tf.maximum(conv_right_left, 0)
            feature_list_new.append(conv_right_left)

    feature_list_new.reverse()
    feature_right_left = tf.stack(feature_list_new, axis=2)
    feature_right_left = tf.squeeze(feature_right_left, axis=3)

    merge_way = 1
    processed_feature = input_feature
    if merge_way == 1:
        # merge 1: concat directly
        merge_feature1 = tf.concat([feature_top_down, feature_down_top,
                                    feature_left_right, feature_right_left], axis=3)
        merge_feature1 = slim.conv2d(merge_feature1, input_feature.get_shape().as_list()[3], [1, 1], activation_fn=None)
        processed_feature = merge_feature1

    elif merge_way == 2:
        # merge 2: concat and residual connect
        merge_feature2 = tf.concat([feature_top_down, feature_down_top,
                                    feature_left_right, feature_right_left], axis=3)
        merge_feature2 = slim.conv2d(merge_feature2, input_feature.get_shape().as_list()[3], [1, 1], activation_fn=None)
        merge_feature2 = merge_feature2 + input_feature
        processed_feature = merge_feature2

    elif merge_way == 3:
        # merge 3: direct-aware attention map, sigmoid, concat and mul
        att_top_down = slim.conv2d(feature_top_down, 1, [1, 1], activation_fn=nn.sigmoid)
        att_top_down_feature = tf.multiply(input_feature, att_top_down)
        att_down_top = slim.conv2d(feature_down_top, 1, [1, 1], activation_fn=nn.sigmoid)
        att_down_top_feature = tf.multiply(input_feature, att_down_top)
        att_left_right = slim.conv2d(feature_left_right, 1, [1, 1], activation_fn=nn.sigmoid)
        att_left_right_feature = tf.multiply(input_feature, att_left_right)
        att_right_left = slim.conv2d(feature_right_left, 1, [1, 1], activation_fn=nn.sigmoid)
        att_right_left_feature = tf.multiply(input_feature, att_right_left)
        merge_feature3 = tf.concat([att_top_down_feature, att_down_top_feature,
                                    att_left_right_feature, att_right_left_feature], axis=3)
        merge_feature3 = slim.conv2d(merge_feature3, input_feature.get_shape().as_list()[3], [1, 1], activation_fn=None)
        processed_feature = merge_feature3

    return processed_feature

def atrous_spp(input_feature, depth=256):      # c: 256
    '''
    aspp for deeplabv3, output_stride=16, [6, 12, 18]; output_stide=8, rate:[12, 24, 36]
    :param input_feature:
    :param k: kernel size: 1xk, kx1
    :return: feature
    '''

    # 1x1 conv
    at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

    # rate = 6
    at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=12, activation_fn=None)

    # rate = 12
    at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=24, activation_fn=None)

    # rate = 18
    at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=36, activation_fn=None)

    # image pooling
    img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
    img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
    img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                         input_feature.get_shape().as_list()[2]))

    net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                    axis=3, name='atrous_concat')
    net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

    return net

def atrous_spp16(input_feature, depth=256):      # c: 256
    '''
    aspp for deeplabv3, output_stride=16, [6, 12, 18]; output_stide=8, rate:[12, 24, 36]
    :param input_feature:
    :param k: kernel size: 1xk, kx1
    :return: feature
    '''
    with tf.variable_scope("aspp"):
        # 1x1 conv
        at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

        # rate = 6
        at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=6, activation_fn=None)

        # rate = 12
        at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=12, activation_fn=None)

        # rate = 18
        at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=18, activation_fn=None)

        # image pooling
        img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
        img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
        img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                             input_feature.get_shape().as_list()[2]))

        net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                        axis=3, name='atrous_concat')
        net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

        return net

def conv(input_feat, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding='VALID', biased=True, is_training=True):
    c_i = input_feat.get_shape()[-1]
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable('weights', shape=[k_h, k_w, c_i, c_o], trainable=is_training)
        output = convolve(input_feat, kernel)

        if biased:
            biases = tf.get_variable('biases', [c_o], trainable=is_training)
            output = tf.nn.bias_add(output, biases)
        if relu:
            output = tf.nn.relu(output, name=scope.name)
        return output

def psp_module(input_feature, depth=512):
    '''psp model'''

    shape = input_feature.get_shape().as_list()
    def avg_pool_conv_resize(input_feature, k_h, k_w, s_h, s_w, name):
        pool = tf.nn.avg_pool(input_feature,
                                ksize=[1, k_h, k_w, 1],
                                strides=[1, s_h, s_w, 1],
                                padding='VALID',
                                name=name)
        down_dim = conv(pool, 1, 1, int(shape[3]/4), 1, 1, name=name + '_conv', biased=False, relu=False)
        # BN ?
        output = tf.image.resize_images(down_dim, [shape[1], shape[2]])
        return output

    pool1_interp = avg_pool_conv_resize(input_feature, 60, 60, 60, 60, name='pool1')
    pool2_interp = avg_pool_conv_resize(input_feature, 30, 30, 30, 30, name='pool2')
    pool3_interp = avg_pool_conv_resize(input_feature, 20, 20, 20, 20, name='pool3')
    pool6_interp = avg_pool_conv_resize(input_feature, 10, 10, 10, 10, name='pool6')

    concat = tf.concat([input_feature, pool6_interp, pool3_interp, pool2_interp, pool1_interp], axis=3)
    output = conv(concat, 3, 3, depth, 1, 1, name='concat_down_dim', biased=False, relu=False, padding='SAME')

    return output

def channel_attention(input_feature, name):
    '''
    :param x: b x h x w x c
    :param name:
    :return:
    '''
    depth = input_feature.get_shape().as_list()[3]
    # Global pooling
    global_pooling = tf.reduce_mean(input_feature, [1, 2], name=name + '/image_level_global_pooling', keep_dims=True)
    fc1 = slim.conv2d(global_pooling, depth, [1, 1], scope=name+'/fc1', normalizer_fn=None)
    fc2 = slim.conv2d(fc1, depth, [1, 1], scope=name+'/fc2', activation_fn=None, normalizer_fn=None)
    sigmoid = tf.nn.sigmoid(fc2, 'channel_attention_sigmoid')

    output_feature = input_feature * sigmoid
    return output_feature

def global_local_att(input_feature):
    '''
    global local attention for bottom of facade image
    :param input_feature:
    :return: feature
    '''

    depth = input_feature.get_shape().as_list()[3]
    size = input_feature.get_shape().as_list()
    # global attention
    input_feature = rnn_parsing_continue(input_feature, 1, depth)

    # bidirectional LSTM

    # attention mul
    kernel = input_feature  # LSTM
    kernel = tf.nn.softmax(kernel, dim=3)
    x = tf.pad(input_feature, paddings=0)
    x = tf.reshape(x, [size[0], size[1], 10 * 10])
    kernel = tf.reshape(kernel, [size[0], 10 * 10, -1])

    x = tf.multiply(x, kernel)
    x = tf.reshape(x, size)

def local_att(input_feature):
    '''

    :param input_feature:
    :return:
    '''
    # local attention:
    pad = 3
    size = input_feature.get_shape().as_list()
    local_feature = slim.conv2d(input_feature, 128, [7, 7], rate=2, scope='local_att')
    local_att = slim.conv2d(local_feature, 49, [1, 1], scope='local_att_trans')
    local_feature_softmax = tf.nn.softmax(local_att, dim=3)

    kernel = tf.reshape(local_feature_softmax, [size[0], 1, size[2] * size[3], 7 * 7])

    print('Before unfold', input_feature.shape())
    x = tf.pad(input_feature, paddings=6)
    print('After unfold', x)
    x = tf.reshape(x, [size[0], size[1], size[2] * size[3], -1])
    print(x.shape(), kernel.shape)

    x = tf.multiply(x, kernel)
    x = tf.reduce_sum(x, keep_dims=3)
    x = tf.reshape(x, size)

    return x

def large_kernel3(x, c, k, r, name):
    '''
    large kernel for facade
    :param x:  input feature
    :param c: output channel
    :param k: kernel size
    :param r: rate for conv
    :return:
    '''
    r2 = int(r*2)
    # 1xk + kx1
    left_1 = slim.conv2d(x, c, [1, k], scope=name + '/left_1', rate=r)
    left_2 = slim.conv2d(left_1, c, [k, 1], scope=name + '/left_2', rate=r)
    left_3 = slim.conv2d(left_2, c, [1, k], scope=name + '/left_3', rate=r2)
    left_4 = slim.conv2d(left_3, c, [k, 1], scope=name + '/left_4', rate=r2)

    right_1 = slim.conv2d(x, c, [k, 1], scope=name + '/right_1',rate=r)
    right_2 = slim.conv2d(right_1, c, [1, k], scope=name + '/right_2', rate=r)
    right_3 = slim.conv2d(right_2, c, [k, 1], scope=name + '/right_3',rate=r2)
    right_4 = slim.conv2d(right_3, c, [1, k], scope=name + '/right_4', rate=r2)

    y = left_4 + right_4

    return y

def large_kernel4(x, c, k, r, name):
    '''
    large kernel for facade
    :param x:  input feature
    :param c: output channel
    :param k: kernel size
    :param r: rate for conv
    :return:
    '''
    r2 = int(r * 2)
    r3 = int(r2 * 2)
    # 1xk + kx1
    left_1 = slim.conv2d(x, c, [1, k], scope=name + '/left_1', rate=r)
    left_2 = slim.conv2d(left_1, c, [k, 1], scope=name + '/left_2', rate=r)
    left_3 = slim.conv2d(left_2, c, [1, k], scope=name + '/left_3', rate=r2)
    left_4 = slim.conv2d(left_3, c, [k, 1], scope=name + '/left_4', rate=r2)
    left_5 = slim.conv2d(left_4, c, [1, k], scope=name + '/left_5', rate=r3)
    left_6 = slim.conv2d(left_5, c, [k, 1], scope=name + '/left_6', rate=r3)

    right_1 = slim.conv2d(x, c, [k, 1], scope=name + '/right_1',rate=r)
    right_2 = slim.conv2d(right_1, c, [1, k], scope=name + '/right_2', rate=r)
    right_3 = slim.conv2d(right_2, c, [k, 1], scope=name + '/right_3',rate=r2)
    right_4 = slim.conv2d(right_3, c, [1, k], scope=name + '/right_4', rate=r2)
    right_5 = slim.conv2d(right_4, c, [k, 1], scope=name + '/right_5', rate=r3)
    right_6 = slim.conv2d(right_5, c, [1, k], scope=name + '/right_6', rate=r3)

    y = left_6 + right_6

    return y

def large_kernel2(x, c, k, r, name):
    '''
    large kernel for facade
    :param x:  input feature
    :param c: output channel
    :param k: kernel size
    :param r: rate for conv
    :return:
    '''
    # 1xk + kx1
    left_1 = slim.conv2d(x, c, [1, k], scope=name + '/left_1', rate=r)
    left = slim.conv2d(left_1, c, [k, 1], scope=name + '/left_2', rate=r)

    right_1 = slim.conv2d(x, c, [k, 1], scope=name + '/right_1',rate=r)
    right = slim.conv2d(right_1, c, [1, k], scope=name + '/right_2', rate=r)

    y = left + right

    return y

def large_kernel1(x, c, k, r, name):
    '''
    large kernel for facade
    :param x:  input feature
    :param c: output channel
    :param k: kernel size
    :param r: rate for conv
    :return:
    '''
    # 1xk + kx1
    row = slim.conv2d(x, c, [1, k], scope=name + '/row', rate=r)
    col = slim.conv2d(x, c, [k, 1], scope=name + '/col', rate=r)
    y = row + col
    return y

def fcn_backbone(image, is_training=True):
    with tf.variable_scope('vgg_16'):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu):
            conv1 = slim.repeat(image, 2, slim.conv2d, 64, [3, 3],
                                trainable=is_training, scope='conv1')
            pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME', scope='pool1')
            conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3],
                                trainable=is_training, scope='conv2')
            pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME', scope='pool2')
            conv3 = slim.repeat(pool2, 3, slim.conv2d, 256, [3, 3],
                                trainable=is_training, scope='conv3')
            pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME', scope='pool3')
            conv4 = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3],
                                trainable=is_training, scope='conv4')
            pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME', scope='pool4')
            conv5 = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3],
                                trainable=is_training, scope='conv5')
            pool5 = slim.max_pool2d(conv5, [2, 2], padding='SAME', scope='pool5')

            # VGG fc6 and fc7
            fc6 = slim.conv2d(pool5, 4096, [7, 7], scope='fc6', trainable=is_training, rate=4)
            fc6 = slim.dropout(fc6, 0.5, is_training=is_training, scope='dropout6')
            fc7 = slim.conv2d(fc6, 4096, [1, 1], scope='fc7', trainable=is_training)
            net = fc7

            pool4_shape = pool4.get_shape().as_list()
            pool4_up = tf.image.resize_images(fc7, [pool4_shape[1], pool4_shape[2]])

            merge1 = tf.concat([pool4, pool4_up], axis=3)
            merge1 = slim.conv2d(merge1, 256, [1, 1], scope='merge1', trainable=is_training)

            pool3_shape = pool3.get_shape().as_list()
            pool3_up = tf.image.resize_images(merge1, [pool3_shape[1], pool3_shape[2]])

            merge2 = tf.concat([pool3, pool3_up], axis=3)


            end_points = {'conv1': conv1,
                          'conv2': conv2,
                          'conv3': conv3,
                          'conv4': conv4,
                          'conv5': conv5}
    return merge2, end_points

def group_norm(x, G=32, eps=1e-5, scope='group_norm') :
    with tf.variable_scope(scope) :
        N, H, W, C = x.get_shape().as_list()
        G = min(G, C)

        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)

        gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))


        x = tf.reshape(x, [N, H, W, C]) * gamma + beta

    return x

def object_context_attention(x, key_channels=256, value_channels=256, out_channels=512):
    ''''''
    batch_size, h, w, c = x.get_shape().as_list()
    query = slim.conv2d(x, key_channels, [1, 1], scope='query', activation_fn=None)
    query = tf.transpose(query, [0, 3, 1, 2])
    query = tf.reshape(query, [batch_size, key_channels, -1])
    query = tf.transpose(query, [0, 2, 1])

    key = slim.conv2d(x, key_channels, [1, 1], scope='key', activation_fn=None)
    key = tf.transpose(key, [0, 3, 1, 2])
    key = tf.reshape(key, [batch_size, key_channels, -1])


    value = slim.conv2d(x, value_channels, [1, 1], scope='value', activation_fn=None)
    value = tf.transpose(value, [0, 3, 1, 2])
    value = tf.reshape(value, [batch_size, value_channels, -1])
    value = tf.transpose(value, [0, 2, 1])

    similirity_map = tf.matmul(query, key, name='similirity_map')
    similirity_map = (key_channels**-.5) * similirity_map
    similirity_map = tf.nn.softmax(similirity_map, dim=-1)

    context = tf.matmul(similirity_map, value)
    context = tf.transpose(context, [0, 2, 1])
    context = tf.reshape(context, [batch_size, h, w, value_channels])


    context = slim.conv2d(context, out_channels, [1, 1], scope='out', activation_fn=None)

    out = tf.concat([context, x], axis=3)
    out = slim.conv2d(out, out_channels, [1, 1], scope='concat_dim', activation_fn=None)

    return out

def large_kernel_aspp(x, c, name):
    with tf.variable_scope('large_kernel_aspp'):
        with tf.variable_scope('k1'):
            k1 = 3
            r1 = 4
            left_1 = slim.conv2d(x, c, [1, k1], scope=name + '/left_1', rate=r1)
            left = slim.conv2d(left_1, c, [k1, 1], scope=name + '/left_2', rate=r1)

            right_1 = slim.conv2d(x, c, [k1, 1], scope=name + '/right_1', rate=r1)
            right = slim.conv2d(right_1, c, [1, k1], scope=name + '/right_2', rate=r1)
        y1 = left + right

        with tf.variable_scope('k2'):
            k1 = 7
            r1 = 4
            left_1 = slim.conv2d(x, c, [1, k1], scope=name + '/left_1', rate=r1)
            left = slim.conv2d(left_1, c, [k1, 1], scope=name + '/left_2', rate=r1)

            right_1 = slim.conv2d(x, c, [k1, 1], scope=name + '/right_1', rate=r1)
            right = slim.conv2d(right_1, c, [1, k1], scope=name + '/right_2', rate=r1)
        y2 = left + right

        with tf.variable_scope('k3'):
            k1 = 11
            r1 = 4
            left_1 = slim.conv2d(x, c, [1, k1], scope=name + '/left_1', rate=r1)
            left = slim.conv2d(left_1, c, [k1, 1], scope=name + '/left_2', rate=r1)

            right_1 = slim.conv2d(x, c, [k1, 1], scope=name + '/right_1', rate=r1)
            right = slim.conv2d(right_1, c, [1, k1], scope=name + '/right_2', rate=r1)
        y3 = left + right

        with tf.variable_scope('k4'):
            k1 = 15
            r1 = 4
            left_1 = slim.conv2d(x, c, [1, k1], scope=name + '/left_1', rate=r1)
            left = slim.conv2d(left_1, c, [k1, 1], scope=name + '/left_2', rate=r1)

            right_1 = slim.conv2d(x, c, [k1, 1], scope=name + '/right_1', rate=r1)
            right = slim.conv2d(right_1, c, [1, k1], scope=name + '/right_2', rate=r1)
        y4 = left + right

        y = tf.concat([x, y1, y2, y3, y4], axis=3)
        y = slim.conv2d(y, c, [1, 1], scope='conv_1x1_output', activation_fn=None)

    return y

def decoder_block(x, tar_c):
    b, h, w, c = x.get_shape().as_list()
    with tf.variable_scope('upsample'):
        x = slim.conv2d(x, tar_c, [1, 1], scope='down_dim')
        x = slim.conv2d_transpose(x, tar_c, [3, 3], stride=2, scope='conv_trans')
        x = slim.conv2d(x, tar_c, [1, 1], scope='up_dim')
    return x

def feature_decoder(x, end_points):
    '''feature docoder
    x: input feature
    end_points: point feature of conv layers
    '''
    with tf.variable_scope('up1'):
        up1 = decoder_block(x, 256)
        merge1 = tf.concat([end_points['conv3'], up1], axis=3)
        up1 = slim.repeat(merge1, 2, slim.conv2d, 256, [3, 3], scope='conv_up1')

    with tf.variable_scope('up2'):
        up2 = decoder_block(up1, 128)
        merge2 = tf.concat([end_points['conv2'], up2], axis=3)
        up2 = slim.repeat(merge2, 2, slim.conv2d, 128, [3, 3], scope='conv_up2')

    with tf.variable_scope('up3'):
        up3 = decoder_block(up2, 64)
        merge3 = tf.concat([end_points['conv1'], up3], axis=3)
        up3 = slim.repeat(merge3, 2, slim.conv2d, 64, [3, 3], scope='conv_up3')

    net = up3
    return net

def gen_semantic_feature(x):
    b, h, w, c = x.get_shape().as_list()
    with tf.variable_scope('gen_semantic_feature'):
        turn = 4
        x_in = x
        conv_list = []
        for i in range(turn):
            conv3x3_1 = slim.conv2d(x_in, c, [3, 3], scope='conv' + str(i+1))
            # conv3x3_1 = large_kernel1(x_in, c, int(2*(turn-i)+1), 1, 'conv' + str(i+1))
            pooling5x5 = slim.max_pool2d(conv3x3_1, [5, 5], padding='SAME', scope='pool' + str(i+1))
            conv3x3_2 = slim.conv2d(pooling5x5, int(c / turn), [3, 3], scope='conv_down_dim' + str(i+1))
            # conv3x3_2 = large_kernel1(pooling5x5, int(c / turn), int(2 * (turn - i) + 1), 1, 'conv' + str(i + 1))
            up_feature = tf.image.resize_images(conv3x3_2, [h, w])
            conv_list.append(up_feature)
            x_in = pooling5x5
        conv_list.append(x)
        merge = tf.concat(conv_list, axis=3)
        y = slim.conv2d(merge, c, [3, 3], scope='output')
    return y

def denseASPP(net):
    d_feature0 = 512
    d_feature1 = 128

    def _DenseAsppBlock(layer, num1, num2, dilation_rate, keep_prob, scope, bn_start=True):
        with tf.variable_scope(scope, 'ASPP', [layer]) as sc:
            if bn_start:
                layer = slim.batch_norm(layer, scope="norm1")
            layer = tf.nn.relu(layer, name="relu1")
            layer = slim.conv2d(layer, num1, kernel_size=1, scope='conv1')
            layer = slim.batch_norm(layer, scope="norm2")
            layer = tf.nn.relu(layer, name="relu2")
            layer = tf.pad(layer, [[0, 0], [dilation_rate, dilation_rate], [dilation_rate, dilation_rate], [0, 0]])
            layer = slim.conv2d(layer, num2, kernel_size=3, scope='conv2', rate=dilation_rate, padding="VALID")
            layer = slim.dropout(layer, keep_prob=keep_prob)
        return layer

    with tf.variable_scope('ASPP', 'ASPP') as sc1:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=False), \
             slim.arg_scope([slim.batch_norm],
                            epsilon=1e-5, scale=True,fused=False), \
             slim.arg_scope([slim.batch_norm, slim.conv2d],
                            trainable=False, activation_fn=None):

                net = slim.conv2d(net, 1024, [1, 1], scope='down_dim', activation_fn=None)
                net = slim.batch_norm(net, scope="norm")

                ASPP_3 = _DenseAsppBlock(net, num1=d_feature0, num2=d_feature1, dilation_rate=3, keep_prob=1,
                                         scope="ASPP_3", bn_start=False)
                net = tf.concat([ASPP_3, tf.nn.relu(net)], axis=-1)

                ASPP_6 = _DenseAsppBlock(net, num1=d_feature0, num2=d_feature1, dilation_rate=6, keep_prob=1,
                                         scope="ASPP_6", bn_start=True)
                net = tf.concat([ASPP_6, net], axis=-1)

                ASPP_12 = _DenseAsppBlock(net, num1=d_feature0, num2=d_feature1, dilation_rate=12, keep_prob=1,
                                          scope="ASPP_12", bn_start=True)
                net = tf.concat([ASPP_12, net], axis=-1)

                ASPP_18 = _DenseAsppBlock(net, num1=d_feature0, num2=d_feature1, dilation_rate=18, keep_prob=1,
                                          scope="ASPP_18", bn_start=True)
                net = tf.concat([ASPP_18, net], axis=-1)

                ASPP_24 = _DenseAsppBlock(net, num1=d_feature0, num2=d_feature1, dilation_rate=24, keep_prob=1,
                                          scope="ASPP_24", bn_start=True)
                net = tf.concat([ASPP_24, net], axis=-1)

    return net

def pyramid_module(input_feature, name, depth=256):      # c: 256
    '''
    large kernel for mid of facade img
    :param input_feature:
    :param k: kernel size: 1xk, kx1
    :return: feature
    '''

    # rate = 4
    at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope=name + '/conv_3x3_1', rate=4, activation_fn=None)

    # rate = 8
    at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope=name + '/conv_3x3_2', rate=8, activation_fn=None)

    # rate = 12
    at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope=name + '/conv_3x3_3', rate=12, activation_fn=None)
    #
    # # rate = 16
    # at_pooling3x3_4 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_4', rate=16, activation_fn=None)
    #
    # # rate = 20
    # at_pooling3x3_5 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_5', rate=20, activation_fn=None)

    # # Add spatial msg parsing
    # with tf.variable_scope('spatial_parsing'):
    #     input_feature = spatial_cnn(input_feature, k=9, c=256)
    # with tf.variable_scope('spatial_parsing', reuse=True):
    #     at_pooling3x3_1 = spatial_cnn(at_pooling3x3_1, k=9, c=256)
    # with tf.variable_scope('spatial_parsing', reuse=True):
    #     at_pooling3x3_2 = spatial_cnn(at_pooling3x3_2, k=9, c=256)
    # with tf.variable_scope('spatial_parsing', reuse=True):
    #     at_pooling3x3_3 = spatial_cnn(at_pooling3x3_3, k=9, c=256)
    # with tf.variable_scope('spatial_parsing', reuse=True):
    #     at_pooling3x3_4 = spatial_cnn(at_pooling3x3_4, k=9, c=256)
    # with tf.variable_scope('spatial_parsing', reuse=True):
    #     at_pooling3x3_5 = spatial_cnn(at_pooling3x3_5, k=9, c=256)

    net = tf.concat([input_feature, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                    axis=3, name=name + '/atrous_concat')
    net = slim.conv2d(net, depth, [1, 1], scope=name + '/conv_1x1_output', activation_fn=None)

    return net

def resnet_backbone(image, is_training=True):
    from resnet import resnet_v1
    _BATCH_NORM_DECAY = 0.9997

    with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope(batch_norm_decay=_BATCH_NORM_DECAY)):
        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=is_training,
                                                 global_pool=False, output_stride=8,
                                                 spatial_squeeze=False)
    return net, end_points

def resnet_backbone101(image, is_training=True):
    from resnet import resnet_v1
    with slim.arg_scope(
            resnet_v1.resnet_arg_scope()):  # , normalizer_fn=slim.batch_norm, normalizer_params={'is_training': False}):

        net, end_points = resnet_v1.resnet_v1_101(image, num_classes=None, is_training=None,  # is_training: None: not trian BN paras
                                                 global_pool=False, output_stride=8,
                                                 spatial_squeeze=False)
        end = {'conv1':end_points['resnet_v1_101/conv1'],
            'conv2':end_points['resnet_v1_101/block1/unit_3/bottleneck_v1'],
            'conv3':end_points['resnet_v1_101/block2/unit_4/bottleneck_v1'],
            'conv4': end_points['resnet_v1_101/block3/unit_23/bottleneck_v1'],
            'conv5': end_points['resnet_v1_101/block4/unit_3/bottleneck_v1'],}
    return net, end

def resnet_backbone152(image, is_training=True):
    from resnet import resnet_v1
    with slim.arg_scope(
            resnet_v1.resnet_arg_scope()):  # , normalizer_fn=slim.batch_norm, normalizer_params={'is_training': False}):

        net, end_points = resnet_v1.resnet_v1_152(image, num_classes=None, is_training=None,  # is_training: None: not trian BN paras
                                                 global_pool=False, output_stride=8,
                                                 spatial_squeeze=False)
        end = {'conv1':end_points['resnet_v1_101/conv1'],
            'conv2':end_points['resnet_v1_101/block1/unit_3/bottleneck_v1'],
            'conv3':end_points['resnet_v1_101/block2/unit_8/bottleneck_v1'],
            'conv4': end_points['resnet_v1_101/block3/unit_36/bottleneck_v1'],
            'conv5': end_points['resnet_v1_101/block4/unit_3/bottleneck_v1'],}
    return net, end

def vgg16_backbone(image, is_training=True):
    with tf.variable_scope('vgg_16'):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu):
            conv1 = slim.repeat(image, 2, slim.conv2d, 64, [3, 3],
                                trainable=is_training, scope='conv1')
            pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME', scope='pool1')
            conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3],
                                trainable=is_training, scope='conv2')
            pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME', scope='pool2')
            conv3 = slim.repeat(pool2, 3, slim.conv2d, 256, [3, 3],
                                trainable=is_training, scope='conv3')
            pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME', scope='pool3')
            conv4 = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3],
                                trainable=is_training, scope='conv4')

            # add dilated convolution ###
            conv5_1 = slim.conv2d(conv4, 512, [3, 3], scope='conv5/conv5_1', trainable=is_training, rate=2)
            conv5_2 = slim.conv2d(conv5_1, 512, [3, 3], scope='conv5/conv5_2', trainable=is_training, rate=2)
            conv5_3 = slim.conv2d(conv5_2, 512, [3, 3], scope='conv5/conv5_3', trainable=is_training, rate=2)

            end_points = {'conv1': conv1,
                            'conv2': conv2,
                            'conv3': conv3,
                            'conv4': conv4,
                            'conv5': conv5_3}
    net = conv5_3
    return net, end_points

# ------------------------Network---------------------------------

def inference_fcn(image, is_training):
    with tf.variable_scope('vgg_16'):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu):
            conv1 = slim.repeat(image, 2, slim.conv2d, 64, [3, 3],
                                trainable=is_training, scope='conv1')
            pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME', scope='pool1')
            conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3],
                                trainable=is_training, scope='conv2')
            pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME', scope='pool2')
            conv3 = slim.repeat(pool2, 3, slim.conv2d, 256, [3, 3],
                                trainable=is_training, scope='conv3')
            pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME', scope='pool3')
            conv4 = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3],
                                trainable=is_training, scope='conv4')
            pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME', scope='pool4')
            conv5 = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3],
                                trainable=is_training, scope='conv5')
            pool5 = slim.max_pool2d(conv5, [2, 2], padding='SAME', scope='pool5')

            fc6 = slim.conv2d(pool5, 4096, [7, 7], scope='fc6', trainable=is_training)
            dropout6 = slim.dropout(fc6, 0.85)
            fc7 = slim.conv2d(dropout6, 4096, [1, 1], scope='fc7', trainable=is_training)
            dropout7 = slim.dropout(fc7, 0.85)

            W8 = weight_variable([1, 1, 4096, cfg.NUM_OF_CLASSESS], name="W8")
            b8 = bias_variable([cfg.NUM_OF_CLASSESS], name="b8")
            conv8 = conv2d_basic(dropout7, W8, b8)
            # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

            # now to upscale to actual image size
            deconv_shape1 = pool4.get_shape()
            W_t1 = weight_variable([4, 4, deconv_shape1[3].value, cfg.NUM_OF_CLASSESS], name="W_t1")
            b_t1 = bias_variable([deconv_shape1[3].value], name="b_t1")
            conv_t1 = conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(pool4))
            fuse_1 = tf.add(conv_t1, pool4, name="fuse_1")

            deconv_shape2 = pool3.get_shape()
            W_t2 = weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
            b_t2 = bias_variable([deconv_shape2[3].value], name="b_t2")
            conv_t2 = conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(pool3))
            fuse_2 = tf.add(conv_t2, pool3, name="fuse_2")

            shape = tf.shape(image)
            deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], cfg.NUM_OF_CLASSESS])
            W_t3 = weight_variable([16, 16, cfg.NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
            b_t3 = bias_variable([cfg.NUM_OF_CLASSESS], name="b_t3")
            conv_t3 = conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

            annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

        return tf.expand_dims(annotation_pred, dim=3), conv_t3, fuse_2

def inference_Unet(image, is_training):
    '''Unet '''
    with tf.variable_scope('Unet'):
        conv1 = slim.repeat(image, 2, slim.conv2d, 64, [3, 3],
                                trainable=is_training, scope='conv1')
        pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME', scope='pool1')
        conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3],
                            trainable=is_training, scope='conv2')
        pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME', scope='pool2')
        conv3 = slim.repeat(pool2, 2, slim.conv2d, 256, [3, 3],
                            trainable=is_training, scope='conv3')
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME', scope='pool3')
        conv4 = slim.repeat(pool3, 2, slim.conv2d, 512, [3, 3],
                            trainable=is_training, scope='conv4')
        drop4 = slim.dropout(conv4, keep_prob=0.5)
        pool4 = slim.max_pool2d(drop4, [2, 2], padding='SAME', scope='pool4')

        conv5 = slim.repeat(pool4, 2, slim.conv2d, 1024, [3, 3],
                            trainable=is_training, scope='conv5')
        drop5 = slim.dropout(conv5, keep_prob=0.5)


        up6 = slim.conv2d_transpose(drop5, 512, [2, 2], stride=2,
                                        scope='up6', trainable=is_training)
        merge6 = tf.concat([drop4, up6], axis=3)
        conv6 = slim.repeat(merge6, 2, slim.conv2d, 512, [3, 3],
                            trainable=is_training, scope='conv6')

        up7 = slim.conv2d_transpose(conv6, 256, [2, 2], stride=2,
                                        scope='up7', trainable=is_training)
        merge7 = tf.concat([conv3, up7], axis=3)
        conv7 = slim.repeat(merge7, 2, slim.conv2d, 256, [3, 3],
                            trainable=is_training, scope='conv7')

        up8 = slim.conv2d_transpose(conv7, 128, [2, 2], stride=2,
                                        scope='up8', trainable=is_training)
        merge8 = tf.concat([conv2, up8], axis=3)
        conv8 = slim.repeat(merge8, 2, slim.conv2d, 128, [3, 3],
                            trainable=is_training, scope='conv8')

        up9 = slim.conv2d_transpose(conv8, 64, [2, 2], stride=2,
                                    scope='up9', trainable=is_training)
        merge9 = tf.concat([conv1, up9], axis=3)
        conv9 = slim.repeat(merge9, 2, slim.conv2d, 64, [3, 3],
                            trainable=is_training, scope='conv9')


        outputs = slim.conv2d(conv9, cfg.NUM_OF_CLASSESS, [1, 1],
                             scope='logits', trainable=is_training, activation_fn=None)

        label_pred = tf.expand_dims(tf.argmax(outputs, axis=3, name="prediction"), dim=3)
    return label_pred, outputs, conv9

def inference_TernausNet(image, is_training):
    '''TernausNet'''
    with tf.variable_scope('vgg_16'):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu):
            conv1 = slim.repeat(image, 2, slim.conv2d, 64, [3, 3],
                                trainable=is_training, scope='conv1')
            pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME', scope='pool1')
            conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3],
                                trainable=is_training, scope='conv2')
            pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME', scope='pool2')
            conv3 = slim.repeat(pool2, 3, slim.conv2d, 256, [3, 3],
                                trainable=is_training, scope='conv3')
            pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME', scope='pool3')
            conv4 = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3],
                                trainable=is_training, scope='conv4')
            pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME', scope='pool4')
            conv5 = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3],
                                trainable=is_training, scope='conv5')
            pool5 = slim.max_pool2d(conv5, [2, 2], padding='SAME', scope='pool5')

            center = slim.conv2d(pool5, 512, [3, 3], scope='center', trainable=is_training)

            # UP1
            up1 = slim.conv2d_transpose(center, 256, [3, 3], stride=2,
                                        scope='up1', trainable=is_training)
            concat1 = tf.concat([up1, conv5], axis=3, name='concat1')
            concat1 = slim.conv2d(concat1, 512, [3, 3], scope='concat1_conv', trainable=is_training)
            # UP2
            up2 = slim.conv2d_transpose(concat1, 256, [3, 3], stride=2,
                                        scope='up2', trainable=is_training)
            concat2 = tf.concat([up2, conv4], axis=3, name='concat2')
            concat2 = slim.conv2d(concat2, 512, [3, 3], scope='concat2_conv', trainable=is_training)
            # UP3
            up3 = slim.conv2d_transpose(concat2, 128, [3, 3], stride=2,
                                        scope='up3', trainable=is_training)
            concat3 = tf.concat([up3, conv3], axis=3, name='concat3')
            concat3 = slim.conv2d(concat3, 256, [3, 3], scope='concat3_conv', trainable=is_training)
            # UP4
            up4 = slim.conv2d_transpose(concat3, 64, [3, 3], stride=2,
                                        scope='up4', trainable=is_training)
            concat4 = tf.concat([up4, conv2], axis=3, name='concat4')
            concat4 = slim.conv2d(concat4, 128, [3, 3], scope='concat4_conv', trainable=is_training)
            # UP5
            up5 = slim.conv2d_transpose(concat4, 32, [3, 3], stride=2,
                                        scope='up5', trainable=is_training)
            concat5 = tf.concat([up5, conv1], axis=3, name='concat5')
            concat5 = slim.conv2d(concat5, 64, [3, 3], scope='concat5_conv', trainable=is_training)

            net = concat5

        fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None)

        img_shape = image.get_shape().as_list()
        output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])

        label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s

def inference_resnet50(image, is_training):
    # net, end_points = vgg16_backbone(image)
    net, end_points = resnet_backbone(image)
    # net, end_points = resnet_backbone101(image)
    # net = resudual_block_channel(net, 256, 'res_down')

    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None,
                       normalizer_fn=None, weights_initializer=init_ops.variance_scaling_initializer)

    img_shape = image.get_shape().as_list()
    logits = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])

    label_pred = tf.expand_dims(tf.argmax(logits, axis=3, name="prediction"), dim=3)
    return label_pred, logits

def inference_resnetv2_50(image, is_training):
    _BATCH_NORM_DECAY = 0.9997
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=_BATCH_NORM_DECAY)):
        net, end_points = resnet_v2.resnet_v2_50(image,
                                                 num_classes=None,
                                                 is_training=False,
                                                 global_pool=False,
                                                 output_stride=8)
    # net, end_points = resnet_backbone101(image)
    # net = resudual_block_channel(net, 256, 'res_down')

    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None,
                       normalizer_fn=None)

    img_shape = image.get_shape().as_list()
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])

    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s, net

def inference_resnet101(image, is_training):
    # net, end_points = vgg16_backbone(image)
    # net, end_points = resnet_backbone(image)
    net, end_points = resnet_backbone101(image)
    net = resudual_block_channel(net, 256, 'res_down')

    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None,
                       normalizer_fn=None)

    img_shape = image.get_shape().as_list()
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])

    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s, net

def inference_resnet101v2(image, is_training):
    _BATCH_NORM_DECAY = 0.9997
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=_BATCH_NORM_DECAY)):
        net, end_points = resnet_v2.resnet_v2_101(image,
                                                  num_classes=None,
                                                  is_training=is_training,
                                                  global_pool=False,
                                                  output_stride=8)

    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None,
                       normalizer_fn=None)

    img_shape = image.get_shape().as_list()
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])

    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s, net

def inference_xception65(image, is_training):
    from resnet.xception import xception_65
    from resnet import xception
    with slim.arg_scope(xception.xception_arg_scope()):
        net, end_points = xception_65(image,
                                      num_classes=None,
                                      is_training=False,
                                      global_pool=False,
                                      output_stride=8)

    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None,
                       normalizer_fn=None)

    img_shape = image.get_shape().as_list()
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])

    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s, net

def inference_mobilenet(image, is_training):
    from resnet import mobile
    net, end_points = mobile._mobilenet_v2(image,
                                           depth_multiplier=1.0,
                                           output_stride=8)

    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None,
                       normalizer_fn=None)

    img_shape = image.get_shape().as_list()
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])

    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s, net

def inference_lfv(image, is_training):

    net, end_points = vgg16_backbone(image)
    # net, end_points = resnet_backbone(image)

    # fc6 = large_kernel(net, 15, 1024, 4, name='large_kernel')

    # 3x3 backbone
    fc6 = slim.conv2d(net, 1024, [3, 3], scope='fc6/fc6', trainable=is_training,
                      rate=12)
    fc7 = slim.conv2d(fc6, 1024, [1, 1], scope='fc7/fc7', trainable=is_training)
    net = fc7

    # larege kernel + aspp
    #fc6 = large_kernel2(net, 512, 15, 4, name='large_kernel')
    #fc7 = slim.conv2d(fc6, 512, [1, 1], scope='fc7/fc7', trainable=is_training)
    #net = fc7
    #net = slim.conv2d(net, 256, [1, 1], scope='down_dim', trainable=is_training)
    #net = pyramid_module(net, 'aspp', depth=256)

    # # Large kernel aspp
    # net = slim.conv2d(net, 256, [1, 1], scope='down_dim', trainable=is_training)
    # net = large_kernel_aspp(net, 256, 'large_kernel_aspp')


    # # # pyramid + aspp
    # net = slim.conv2d(net, 256, [1, 1], scope='down_dim', trainable=is_training)
    # net = pyramid_module(net, depth=256)
    # net = large_kernel2(net, 256, 15, 4, name='large_kernel')

    # # Bi-LSTM
    # net = slim.conv2d(fc7, 512, [1, 1], scope='down_dim_lstm', trainable=is_training)
    # net = BiDirect_LSTM(net, state_size=256) + net

    # # Object context
    # net = slim.conv2d(fc7, 512, [1, 1], scope='down_dim2', trainable=is_training)
    # net = object_context_attention(net)

    # # Spatial CNN
    # net = slim.conv2d(fc7, 256, [1, 1], scope='collect', trainable=is_training)
    # net = spatial_cnn(net, k=3, c=256)
    # if is_training:
    #     net = slim.dropout(net, keep_prob=1.0, is_training=True, scope='dropout_net')

    # # Decoder
    # net = feature_decoder(net, end_points)

    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None)



    img_shape = image.get_shape().as_list()
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])


    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s, net

def inference_renet(image, is_training=True): # 0.943437782204664
    net, end_points = resnet_backbone(image)
    def res_block(x, channel, name=None):
        res = slim.conv2d(x, channel, [1, 1], scope=name + '_1', padding='SAME')
        res = slim.conv2d(res, channel, [3, 3], scope=name + '_2', padding='SAME', activation_fn=None)
        x = slim.conv2d(x, channel, [1, 1], scope=name + '_x', padding='SAME')
        return tf.nn.relu(tf.add(x, res))

    en1 = res_block(net, 512, 'en1_res1')
    en1 = res_block(en1, 256, 'en1_res2')

    net = ReNet(en1, C=256)

    img_shape = image.get_shape().as_list()
    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None,
                       normalizer_fn=None)
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])
    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)

    return label_pred, output_seg_8s, net

def inference_largekernel(image, is_training):

    def global_convolutional_network(x, k, c, name, r=1):
        # 1xk + kx1

        left_1 = slim.conv2d(x, c, [k, 1], scope=name + '/left_1', rate=r)
        left = slim.conv2d(left_1, c, [1, k], scope=name + '/left_2', rate=r)

        right_1 = slim.conv2d(x, c, [1, k], scope=name + '/right_1', rate=r)
        right = slim.conv2d(right_1, c, [k, 1], scope=name + '/right_2', rate=r)

        y = left + right

        return y

    def boundary_refinement(x, name):
        c = x.get_shape().as_list()[3]
        conv1 = slim.conv2d(x, c, [3, 3], scope=name + '_refine_conv1')
        conv2 = slim.conv2d(conv1, c, [3, 3], scope=name + '_refine_conv2', activation_fn=None)
        y = x + conv2

        return y

    from resnet import resnet_v1
    with slim.arg_scope(
            resnet_v1.resnet_arg_scope()):  # , normalizer_fn=slim.batch_norm, normalizer_params={'is_training': False}):

        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 # is_training: None: not trian BN paras
                                                 global_pool=False, output_stride=32,
                                                 spatial_squeeze=False)
        # 50
        end = {'conv1': end_points['resnet_v1_50/conv1'],
               'conv2': end_points['resnet_v1_50/block1/unit_2/bottleneck_v1'],
               'conv3': end_points['resnet_v1_50/block2/unit_3/bottleneck_v1'],
               'conv4': end_points['resnet_v1_50/block3/unit_5/bottleneck_v1'],
               'conv5': end_points['resnet_v1_50/block4/unit_3/bottleneck_v1'], }
        # 152
        # end = {'conv1': end_points['resnet_v1_50/conv1'],
        #        'conv2': end_points['resnet_v1_50/block1/unit_2/bottleneck_v1'],
        #        'conv3': end_points['resnet_v1_50/block2/unit_7/bottleneck_v1'],
        #        'conv4': end_points['resnet_v1_50/block3/unit_35/bottleneck_v1'],
        #        'conv5': end_points['resnet_v1_50/block4/unit_3/bottleneck_v1'], }

        c = cfg.NUM_OF_CLASSESS
        conv5_gcn = global_convolutional_network(end['conv5'], 15, c, 'conv5_gcn')
        conv4_gcn = global_convolutional_network(end['conv4'], 15, c, 'conv4_gcn')
        conv3_gcn = global_convolutional_network(end['conv3'], 15, c, 'conv3_gcn')
        conv2_gcn = global_convolutional_network(end['conv2'], 15, c, 'conv2_gcn')

        conv5_refine = boundary_refinement(conv5_gcn, 'conv5_refine')
        conv4_refine = boundary_refinement(conv4_gcn, 'conv4_refine')
        conv3_refine = boundary_refinement(conv3_gcn, 'conv3_refine')
        conv2_refine = boundary_refinement(conv2_gcn, 'conv2_refine')

        conv5_deconv = slim.conv2d_transpose(conv5_refine, c, [3, 3], stride=2, scope='up_conv5', trainable=is_training)
        merge1 = conv5_deconv + conv4_refine
        merge1_refine = boundary_refinement(merge1, 'merge1_refine')

        merge1_deconv = slim.conv2d_transpose(merge1_refine, c, [3, 3], stride=2, scope='up_merge1', trainable=is_training)
        merge2 = merge1_deconv + conv3_refine
        merge2_refine = boundary_refinement(merge2, 'merge2_refine')

        merge2_deconv = slim.conv2d_transpose(merge2_refine, c, [3, 3], stride=2, scope='up_merge2', trainable=is_training)
        merge3 = merge2_deconv + conv2_refine
        merge3_refine = boundary_refinement(merge3, 'merge3_refine')

        merge3_deconv = slim.conv2d_transpose(merge3_refine, c, [3, 3], stride=2, scope='up_merge3', trainable=is_training)
        merge = boundary_refinement(merge3_deconv, 'merge_refine')

        merge_deconv = slim.conv2d_transpose(merge, c, [3, 3], stride=2, scope='up_merge', trainable=is_training)
        merge_refine = boundary_refinement(merge_deconv, 'merge_refine2')

    output_seg_8s = merge_refine
    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s, merge_deconv

def inference_scnn(image, is_training):

    net, end_points = vgg16_backbone(image)
    # net, end_points = resnet_backbone(image)
    net = Spatial_CNN(net)
    # net = slim.dropout(net, 0.9)
    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None, normalizer_fn=None)

    img_shape = image.get_shape().as_list()
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])

    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s, net

def inference_res_scnn(image, is_training):
    net, end_points = resnet_backbone(image)
    net = Spatial_CNN(net)
    # net = slim.dropout(net, 0.9)
    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None,
                       normalizer_fn=None)

    img_shape = image.get_shape().as_list()
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])

    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s, net

def inference_parsenet(image, is_training):
    # net, end_points = vgg16_backbone(image)
    net, end_points = resnet_backbone(image)
    # net, end_points = resnet_backbone101(image)
    # net = resudual_block_channel(net, 256, 'res_down')

    # image pooling
    depth = 2048
    img_pooling = tf.reduce_mean(net, [1, 2], name='image_level_global_pooling', keep_dims=True)
    img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
    img_pooling = tf.image.resize_bilinear(img_pooling, (net.get_shape().as_list()[1],
                                                         net.get_shape().as_list()[2]))

    net = tf.concat([img_pooling, net], axis=3, name='atrous_concat')


    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None,
                       normalizer_fn=None)

    img_shape = image.get_shape().as_list()
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])

    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s, net

def inference_vgg_pspnet(image, is_training):

    net, end_points = vgg16_backbone(image)
    # net, end_points = resnet_backbone(image)
    net = psp_module(net, depth=512)
    fc8s = conv(net, 1, 1, cfg.NUM_OF_CLASSESS, 1, 1, biased=True, relu=False, name='logits')
    # fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None, normalizer_fn=None)

    img_shape = image.get_shape().as_list()
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])


    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s

def inference_pspnet(image, is_training):

    net, end = resnet_backbone(image)

    # deep supervise
    dsn = end['resnet_v1_50/block3']
    dsn = conv(dsn, 3, 3, 512, 1, 1, name='dsn_down_dim', biased=False, relu=False, padding='SAME')
    dsn = conv(dsn, 1, 1, cfg.NUM_OF_CLASSESS, 1, 1, biased=True, relu=False, name='logits_dsn')
    net = psp_module(net, depth=512)
    fc8s = conv(net, 1, 1, cfg.NUM_OF_CLASSESS, 1, 1, biased=True, relu=False, name='logits')

    img_shape = image.get_shape().as_list()
    logits_dsn = tf.image.resize_images(dsn, [img_shape[1], img_shape[2]])
    logits = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])

    label_pred = tf.expand_dims(tf.argmax(logits, axis=3, name="prediction"), dim=3)
    return label_pred, logits, logits_dsn

def inference_deeplabv3(image, is_training):

    # net, end_points = vgg16_backbone(image)
    net, end_points = resnet_backbone(image)
    net = atrous_spp(net)
    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None, normalizer_fn=None)

    img_shape = image.get_shape().as_list()
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])


    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s, net

def inference_deeplabv3_he(image, is_training):

    # net, end_points = vgg16_backbone(image)
    net, end_points = resnet_backbone(image)

    input_feature = net
    depth = 256
    # 1x1 conv
    at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None,
                                weights_initializer=init_ops.variance_scaling_initializer)

    # rate = 6
    at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=12, activation_fn=None,
                                  weights_initializer=init_ops.variance_scaling_initializer)

    # rate = 12
    at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=24, activation_fn=None,
                                  weights_initializer=init_ops.variance_scaling_initializer)

    # rate = 18
    at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=36, activation_fn=None,
                                  weights_initializer=init_ops.variance_scaling_initializer)

    # image pooling
    img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
    img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None,
                              weights_initializer=init_ops.variance_scaling_initializer)
    img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                         input_feature.get_shape().as_list()[2]))

    net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                    axis=3, name='atrous_concat')
    net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None,
                      weights_initializer=init_ops.variance_scaling_initializer)
    # if is_training:
    #     net=slim.dropout(net, 0.75, is_training=is_training)

    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None,
                       normalizer_fn=None, weights_initializer=init_ops.variance_scaling_initializer)

    img_shape = image.get_shape().as_list()
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])


    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s, net

def inference_deeplabv3_he_bn(image, is_training):

    # net, end_points = vgg16_backbone(image)
    net, end_points = resnet_backbone(image)

    input_feature = net
    depth = 256
    from tensorflow.python.ops import init_ops

    with slim.arg_scope([slim.conv2d], activation_fn=None,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training,
                                           'scale': True, 'decay': 0.9997,
                                           'epsilon': 1e-5, 'updates_collections': None}):
        # 1x1 conv
        at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None,
                                    weights_initializer=init_ops.variance_scaling_initializer)

        # rate = 6
        at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=12, activation_fn=None,
                                      weights_initializer=init_ops.variance_scaling_initializer)

        # rate = 12
        at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=24, activation_fn=None,
                                      weights_initializer=init_ops.variance_scaling_initializer)

        # rate = 18
        at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=36, activation_fn=None,
                                      weights_initializer=init_ops.variance_scaling_initializer)

        # image pooling
        img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
        img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None,
                                  weights_initializer=init_ops.variance_scaling_initializer)
        img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                             input_feature.get_shape().as_list()[2]))

        net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                        axis=3, name='atrous_concat')
        net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None,
                          weights_initializer=init_ops.variance_scaling_initializer)

    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None,
                       normalizer_fn=None, weights_initializer=init_ops.variance_scaling_initializer)

    img_shape = image.get_shape().as_list()
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])


    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s, net

def inference_deeplabv3_plus(image, is_training):
    print('Start Deeplabv3_plus')
    from resnet import resnet_v1
    with slim.arg_scope(
            resnet_v1.resnet_arg_scope()):  # , normalizer_fn=slim.batch_norm, normalizer_params={'is_training': False}):

        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 # is_training: None: not trian BN paras
                                                 global_pool=False, output_stride=8,
                                                 spatial_squeeze=False)
    # ASPP
    aspp = atrous_spp(net)
    with tf.variable_scope('decoder'):
        # Low level
        low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
        low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
        low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

        # Upsample
        net = tf.image.resize_images(aspp, low_level_features_shape)
        net = tf.concat([net, low_level_features], axis=3)
        net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
        net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')

    # Cls
    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None, normalizer_fn=None)
    img_shape = image.get_shape().as_list()
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])


    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s, aspp

def inference_deeplabv3_plus_16(image, is_training):
    print('Start Deeplabv3_plus')
    from resnet import resnet_v1
    with slim.arg_scope(
            resnet_v1.resnet_arg_scope()):  # , normalizer_fn=slim.batch_norm, normalizer_params={'is_training': False}):

        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 # is_training: None: not trian BN paras
                                                 global_pool=False, output_stride=16,
                                                 spatial_squeeze=False)
    # ASPP
    aspp = atrous_spp16(net)
    with tf.variable_scope('decoder'):
        # Low level
        low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
        low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
        low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

        # Upsample
        net = tf.image.resize_images(aspp, low_level_features_shape)
        net = tf.concat([net, low_level_features], axis=3)
        net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
        net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')

    # Cls
    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None, normalizer_fn=None)
    img_shape = image.get_shape().as_list()
    logits = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])


    label_pred = tf.expand_dims(tf.argmax(logits, axis=3, name="prediction"), dim=3)
    return label_pred, logits

def inference_deeplabv3_plus_16_init(image, is_training):
    print('Start Deeplabv3_plus')
    from resnet import resnet_v1
    with slim.arg_scope(
            resnet_v1.resnet_arg_scope()):  # , normalizer_fn=slim.batch_norm, normalizer_params={'is_training': False}):

        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 # is_training: None: not trian BN paras
                                                 global_pool=False, output_stride=16,
                                                 spatial_squeeze=False)
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=init_ops.variance_scaling_initializer):
        # ASPP
        aspp = atrous_spp16(net)
        with tf.variable_scope('decoder'):
            # Low level
            low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
            low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
            low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

            # Upsample
            net = tf.image.resize_images(aspp, low_level_features_shape)
            net = tf.concat([net, low_level_features], axis=3)
            net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')

        # Cls
        fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None, normalizer_fn=None)
    img_shape = image.get_shape().as_list()
    logits = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])


    label_pred = tf.expand_dims(tf.argmax(logits, axis=3, name="prediction"), dim=3)
    return label_pred, logits

def inference_deeplabv3_plus_16_same(image, is_training):
    print('Start Deeplabv3_plus')
    from resnet import resnet_v1
    with slim.arg_scope(
            resnet_v1.resnet_arg_scope()):  # , normalizer_fn=slim.batch_norm, normalizer_params={'is_training': False}):

        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 # is_training: None: not trian BN paras
                                                 global_pool=False, output_stride=16,
                                                 spatial_squeeze=False)

    with slim.arg_scope([slim.conv2d],
                        trainable=is_training,
                        weights_initializer=init_ops.variance_scaling_initializer):
        # ASPP
        aspp = atrous_spp16(net)
        with tf.variable_scope('decoder'):
            # Low level
            low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
            low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
            low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

            # Upsample
            net = tf.image.resize_images(aspp, low_level_features_shape)
            net = tf.concat([net, low_level_features], axis=3)
            net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')

        # Cls
        fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None, normalizer_fn=None)

    img_shape = image.get_shape().as_list()
    logits = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])

    label_pred = tf.expand_dims(tf.argmax(logits, axis=3, name="prediction"), dim=3)
    return label_pred, logits

def inference_deeplabv3_plus_16_bn(image, is_training):
    print('Start Deeplabv3_plus')
    _BATCH_NORM_DECAY = 0.9997

    from resnet import resnet_v1
    with slim.arg_scope(resnet_v1.resnet_arg_scope(batch_norm_decay=_BATCH_NORM_DECAY)):

        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=is_training,
                                                 # is_training: None: not trian BN paras
                                                 global_pool=False, output_stride=16,
                                                 spatial_squeeze=False)

        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            # ASPP
            aspp = atrous_spp16(net)

        with tf.variable_scope('decoder'):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                # Low level
                low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
                low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
                low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

                # Upsample
                net = tf.image.resize_images(aspp, low_level_features_shape)
                net = tf.concat([net, low_level_features], axis=3)
                net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')

                # Cls
                fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None, normalizer_fn=None)
    img_shape = image.get_shape().as_list()
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])


    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s, aspp

def inference_denseASPP(image, is_training):
    print('Start denseASPP')
    from resnet import resnet_v1
    with slim.arg_scope(
            resnet_v1.resnet_arg_scope()):  # , normalizer_fn=slim.batch_norm, normalizer_params={'is_training': False}):

        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 # is_training: None: not trian BN paras
                                                 global_pool=False, output_stride=8,
                                                 spatial_squeeze=False)
    # ASPP
    net = denseASPP(net)

    # Cls
    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training, activation_fn=None, normalizer_fn=None)
    img_shape = image.get_shape().as_list()
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])


    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)
    return label_pred, output_seg_8s, net

def inference_danet(image, is_training):    # image: [h, w, 9]

    # Resnet encoder
    def res_encoder(image):
        from resnet import resnet_v1
        with slim.arg_scope(
                resnet_v1.resnet_arg_scope()):  # , normalizer_fn=slim.batch_norm, normalizer_params={'is_training': False}):

            net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                     global_pool=False, output_stride=8,
                                                     spatial_squeeze=False)
        return net

    def PAM_Module(x):
        """
            inputs :
                x : input feature maps( B X H X W X C)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        gamma = tf.Variable(tf.zeros([1]), name="sa_gamma")
        b, h, w, c = x.get_shape().as_list()
        c_in = c // 8
        proj_query = slim.conv2d(x, c_in, 1, scope='conv_query', activation_fn=None, biases_initializer=None) # b x h x w x c
        proj_query = tf.reshape(proj_query, [b, int(h*w), c_in])   # b x hw x c

        proj_key = slim.conv2d(x, c_in, 1, scope='conv_key', activation_fn=None, biases_initializer=None) # b x h x w x c
        proj_key = tf.transpose(tf.reshape(proj_key, [b, int(h*w), c_in]), [0, 2, 1])   # b x c x hw

        energy = tf.matmul(proj_query, proj_key)    # b x hw x hw
        attention = tf.nn.softmax(energy)
        attention = tf.transpose(attention, [0, 2, 1])

        proj_value = slim.conv2d(x, c, 1, scope='conv_value', activation_fn=None, biases_initializer=None)
        proj_value = tf.transpose(tf.reshape(proj_value, [b, int(h*w), c]), [0, 2, 1])  # b x c x hw

        out = tf.matmul(proj_value, attention)  # b x c x hw
        out = tf.transpose(tf.reshape(out, [b, c, h, w]), [0, 2, 3, 1]) # b x h x w x c

        out = gamma * out + x
        return out

    def CAM_Module(x):
        """
            inputs :
                x : input feature maps( B X H X W X C)
                returns :
                out : attention value + input feature
                attention: B X C X C
        """
        gamma = tf.Variable(tf.zeros([1]), name="sa_gamma")
        b, h, w, c = x.get_shape().as_list()
        proj_query = tf.transpose(tf.reshape(x, [b, int(h*w), c]), [0, 2, 1])    # b x c x hw
        proj_key = tf.reshape(x, [b, int(h*w), c]) # b x hw x c
        energy = tf.matmul(proj_query, proj_key)    # b x c x c

        tile_size = energy.get_shape().as_list()[2]
        energy_new = tf.reduce_max(energy, -1, keep_dims=True)
        energy_new = tf.tile(energy_new, [1, 1, tile_size]) - energy
        attention = tf.nn.softmax(energy_new)
        proj_value = tf.transpose(tf.reshape(x, [b, int(h*w), c]), [0, 2, 1])   # b x c x hw

        out = tf.matmul(attention, proj_value)  # b x c x hw
        out = tf.transpose(tf.reshape(out, [b, c, h, w]), [0, 2, 3, 1])     # b x h x w x c

        out = gamma * out + x
        return out

    net = res_encoder(image)
    b, h, w, in_channel = net.get_shape().as_list()
    inter_channel = in_channel // 4

    feat1 = slim.conv2d(net, inter_channel, 3, scope='sa_down')
    sa_feat = PAM_Module(feat1)
    sa_conv = slim.conv2d(sa_feat, inter_channel, 3, scope='sa_conv')
    # sa_output

    feat2 = slim.conv2d(net, inter_channel, 3, scope='ca_down')
    ca_feat = CAM_Module(feat2)
    ca_conv = slim.conv2d(ca_feat, inter_channel, 3, scope='ca_conv')
    # ca_output

    feat_sum = sa_conv + ca_conv

    # Cls
    fc8s = slim.conv2d(feat_sum, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_rue', trainable=is_training,
                       activation_fn=None, normalizer_fn=None)
    img_shape = image.get_shape().as_list()
    logits = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])
    label_pred = tf.expand_dims(tf.argmax(logits, axis=3, name="prediction"), dim=3)

    # sa_8s = slim.conv2d(sa_conv, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_sa', trainable=is_training,
    #                    activation_fn=None, normalizer_fn=None)
    # logits_sa = tf.image.resize_images(sa_8s, [img_shape[1], img_shape[2]])
    #
    # ca_8s = slim.conv2d(ca_conv, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_ca', trainable=is_training,
    #                    activation_fn=None, normalizer_fn=None)
    # logits_ca = tf.image.resize_images(ca_8s, [img_shape[1], img_shape[2]])

    return label_pred, logits # , logits_sa, logits_ca


# ---------------------------Our Method------------------------------------

def atrous_large_kernel(x, c, k, r, name):
    '''
    atrous large kernel for facade
    :param x:  input feature
    :param c: output channel
    :param k: kernel size
    :param r: atrous rate for convolutions
    :param name: scope name for alk module
    :return:
    '''

    # 1 x k
    left_1 = slim.conv2d(x, c, [1, k], scope=name + '/left_1', rate=r,
                         weights_initializer=init_ops.variance_scaling_initializer)
    # k x 1
    left = slim.conv2d(left_1, c, [k, 1], scope=name + '/left_2', rate=r,
                       weights_initializer=init_ops.variance_scaling_initializer)
    # k x 1
    right_1 = slim.conv2d(x, c, [k, 1], scope=name + '/right_1', rate=r,
                          weights_initializer=init_ops.variance_scaling_initializer)
    # 1 x k
    right = slim.conv2d(right_1, c, [1, k], scope=name + '/right_2', rate=r,
                        weights_initializer=init_ops.variance_scaling_initializer)

    y = left + right

    return y

def inference_Pyramid_ALKNet(image, is_training):

    # Encoder
    net, end_points = resnet_backbone(image)
    # Decoder
    with tf.variable_scope("decoder"):
        # down dimension and build pyramid scales
        py4 = slim.conv2d(net, 512, [1, 1], scope='py4_dim_down', trainable=is_training,
                         weights_initializer=init_ops.variance_scaling_initializer)
        py2 = slim.max_pool2d(py4, [2, 2], padding='SAME', scope='pooling_py2')
        py1 = slim.max_pool2d(py2, [2, 2], padding='SAME', scope='pooling_py1')

        # pyramid atrous large kernel module
        py1_alk = atrous_large_kernel(py1, 256, 15, 1, 'alk_py1')  # 16x16
        py2_alk = atrous_large_kernel(py2, 256, 15, 2, 'alk_py2')  # 32x32
        py4_alk = atrous_large_kernel(py4, 256, 15, 4, 'alk_py4')  # 64x64

        # Fusion
        py2_shape = py2.get_shape().as_list()
        py1_alk = tf.image.resize_bilinear(py1_alk, (py2_shape[1], py2_shape[2]))
        py2_alk = py2_alk + py1_alk

        py4_shape = py4.get_shape().as_list()
        py2_alk = tf.image.resize_bilinear(py2_alk, (py4_shape[1], py4_shape[2]))
        py4_alk = py4_alk + py2_alk

        net_alk = py4_alk
    # Classifier
    img_shape = image.get_shape().as_list()
    fc8s = slim.conv2d(net_alk, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_cls', trainable=is_training,
                       activation_fn=None,normalizer_fn=None,
                       weights_initializer=init_ops.variance_scaling_initializer)
    logits = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])
    label_pred = tf.expand_dims(tf.argmax(logits, axis=3, name="prediction"), dim=3)

    return label_pred, logits

def inference_Pyramid_ALKNet_k13(image, is_training):

    # Encoder
    net, end_points = resnet_backbone(image)
    # Decoder
    with tf.variable_scope("decoder"):
        # down dimension and build pyramid scales
        py4 = slim.conv2d(net, 512, [1, 1], scope='py4_dim_down', trainable=is_training,
                         weights_initializer=init_ops.variance_scaling_initializer)
        py2 = slim.max_pool2d(py4, [2, 2], padding='SAME', scope='pooling_py2')
        py1 = slim.max_pool2d(py2, [2, 2], padding='SAME', scope='pooling_py1')

        # pyramid atrous large kernel module
        py1_alk = atrous_large_kernel(py1, 256, 13, 1, 'alk_py1')  # 16x16
        py2_alk = atrous_large_kernel(py2, 256, 13, 2, 'alk_py2')  # 32x32
        py4_alk = atrous_large_kernel(py4, 256, 13, 4, 'alk_py4')  # 64x64

        # Fusion
        py2_shape = py2.get_shape().as_list()
        py1_alk = tf.image.resize_bilinear(py1_alk, (py2_shape[1], py2_shape[2]))
        py2_alk = py2_alk + py1_alk

        py4_shape = py4.get_shape().as_list()
        py2_alk = tf.image.resize_bilinear(py2_alk, (py4_shape[1], py4_shape[2]))
        py4_alk = py4_alk + py2_alk

        net_alk = py4_alk
    # Classifier
    img_shape = image.get_shape().as_list()
    fc8s = slim.conv2d(net_alk, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_cls', trainable=is_training,
                       activation_fn=None,normalizer_fn=None,
                       weights_initializer=init_ops.variance_scaling_initializer)
    logits = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])
    label_pred = tf.expand_dims(tf.argmax(logits, axis=3, name="prediction"), dim=3)

    return label_pred, logits

def inference_Pyramid_ALKNet_k11(image, is_training):

    # Encoder
    net, end_points = resnet_backbone(image)
    # Decoder
    with tf.variable_scope("decoder"):
        # down dimension and build pyramid scales
        py4 = slim.conv2d(net, 512, [1, 1], scope='py4_dim_down', trainable=is_training,
                         weights_initializer=init_ops.variance_scaling_initializer)
        py2 = slim.max_pool2d(py4, [2, 2], padding='SAME', scope='pooling_py2')
        py1 = slim.max_pool2d(py2, [2, 2], padding='SAME', scope='pooling_py1')

        # pyramid atrous large kernel module
        py1_alk = atrous_large_kernel(py1, 256, 11, 1, 'alk_py1')  # 16x16
        py2_alk = atrous_large_kernel(py2, 256, 11, 2, 'alk_py2')  # 32x32
        py4_alk = atrous_large_kernel(py4, 256, 11, 4, 'alk_py4')  # 64x64

        # Fusion
        py2_shape = py2.get_shape().as_list()
        py1_alk = tf.image.resize_bilinear(py1_alk, (py2_shape[1], py2_shape[2]))
        py2_alk = py2_alk + py1_alk

        py4_shape = py4.get_shape().as_list()
        py2_alk = tf.image.resize_bilinear(py2_alk, (py4_shape[1], py4_shape[2]))
        py4_alk = py4_alk + py2_alk

        net_alk = py4_alk
    # Classifier
    img_shape = image.get_shape().as_list()
    fc8s = slim.conv2d(net_alk, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_cls', trainable=is_training,
                       activation_fn=None,normalizer_fn=None,
                       weights_initializer=init_ops.variance_scaling_initializer)
    logits = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])
    label_pred = tf.expand_dims(tf.argmax(logits, axis=3, name="prediction"), dim=3)

    return label_pred, logits

def inference_Pyramid_ALKNet_two_layer(image, is_training):

    # Encoder
    net, end_points = resnet_backbone(image)
    # Decoder
    with tf.variable_scope("decoder"):
        # down dimension and build pyramid scales
        py4 = slim.conv2d(net, 512, [1, 1], scope='py4_dim_down', trainable=is_training,
                         weights_initializer=init_ops.variance_scaling_initializer)
        py2 = slim.max_pool2d(py4, [2, 2], padding='SAME', scope='pooling_py2')

        # pyramid atrous large kernel module
        py2_alk = atrous_large_kernel(py2, 256, 15, 2, 'alk_py2')  # 32x32
        py4_alk = atrous_large_kernel(py4, 256, 15, 4, 'alk_py4')  # 64x64

        # Fusion
        py4_shape = py4.get_shape().as_list()
        py2_alk = tf.image.resize_bilinear(py2_alk, (py4_shape[1], py4_shape[2]))
        py4_alk = py4_alk + py2_alk

        net_alk = py4_alk
    # Classifier
    img_shape = image.get_shape().as_list()
    fc8s = slim.conv2d(net_alk, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_cls', trainable=is_training,
                       activation_fn=None,normalizer_fn=None,
                       weights_initializer=init_ops.variance_scaling_initializer)
    logits = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])
    label_pred = tf.expand_dims(tf.argmax(logits, axis=3, name="prediction"), dim=3)

    return label_pred, logits

def inference_Pyramid_ALKNet_one_layer(image, is_training):

    # Encoder
    net, end_points = resnet_backbone(image)
    # Decoder
    with tf.variable_scope("decoder"):
        # down dimension and build pyramid scales
        py4 = slim.conv2d(net, 512, [1, 1], scope='py4_dim_down', trainable=is_training,
                         weights_initializer=init_ops.variance_scaling_initializer)


        # pyramid atrous large kernel module
        py4_alk = atrous_large_kernel(py4, 256, 15, 4, 'alk_py4')  # 64x64

        # Fusion
        net_alk = py4_alk
    # Classifier
    img_shape = image.get_shape().as_list()
    fc8s = slim.conv2d(net_alk, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_cls', trainable=is_training,
                       activation_fn=None,normalizer_fn=None,
                       weights_initializer=init_ops.variance_scaling_initializer)
    logits = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])
    label_pred = tf.expand_dims(tf.argmax(logits, axis=3, name="prediction"), dim=3)

    return label_pred, logits

def inference_Pyramid_ALKNet_2feat(image, is_training):

    # Encoder
    net, end_points = resnet_backbone(image)
    # Decoder
    with tf.variable_scope("decoder"):
        # down dimension and build pyramid scales
        py4 = slim.conv2d(net, 512, [1, 1], scope='py4_dim_down', trainable=is_training,
                         weights_initializer=init_ops.variance_scaling_initializer)
        res4 = slim.conv2d(end_points['resnet_v1_50/block3'], 512, [1, 1], scope='res4_dim_down', trainable=is_training,
                         weights_initializer=init_ops.variance_scaling_initializer)
        py4 = py4 + res4

        py2 = slim.max_pool2d(py4, [2, 2], padding='SAME', scope='pooling_py2')
        py1 = slim.max_pool2d(py2, [2, 2], padding='SAME', scope='pooling_py1')

        # pyramid atrous large kernel module
        py1_alk = atrous_large_kernel(py1, 256, 15, 1, 'alk_py1')  # 16x16
        py2_alk = atrous_large_kernel(py2, 256, 15, 2, 'alk_py2')  # 32x32
        py4_alk = atrous_large_kernel(py4, 256, 15, 4, 'alk_py4')  # 64x64

        # Fusion
        py2_shape = py2.get_shape().as_list()
        py1_alk = tf.image.resize_bilinear(py1_alk, (py2_shape[1], py2_shape[2]))
        py2_alk = py2_alk + py1_alk

        py4_shape = py4.get_shape().as_list()
        py2_alk = tf.image.resize_bilinear(py2_alk, (py4_shape[1], py4_shape[2]))
        py4_alk = py4_alk + py2_alk

        net_alk = py4_alk
    # Classifier
    img_shape = image.get_shape().as_list()
    fc8s = slim.conv2d(net_alk, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_cls', trainable=is_training,
                       activation_fn=None,normalizer_fn=None,
                       weights_initializer=init_ops.variance_scaling_initializer)
    logits = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])
    label_pred = tf.expand_dims(tf.argmax(logits, axis=3, name="prediction"), dim=3)

    return label_pred, logits

def inference_Pyramid_ALKNet_3feat(image, is_training):

    # Encoder
    net, end_points = resnet_backbone(image)
    # Decoder
    with tf.variable_scope("decoder"):
        # down dimension and build pyramid scales
        py4 = slim.conv2d(net, 512, [1, 1], scope='py4_dim_down', trainable=is_training,
                         weights_initializer=init_ops.variance_scaling_initializer)
        res4 = slim.conv2d(end_points['resnet_v1_50/block3'], 512, [1, 1], scope='res4_dim_down', trainable=is_training,
                           weights_initializer=init_ops.variance_scaling_initializer)
        res3 = slim.conv2d(end_points['resnet_v1_50/block2'], 512, [1, 1], scope='res3_dim_down', trainable=is_training,
                           weights_initializer=init_ops.variance_scaling_initializer)
        py4 = py4 + res4 + res3

        py2 = slim.max_pool2d(py4, [2, 2], padding='SAME', scope='pooling_py2')
        py1 = slim.max_pool2d(py2, [2, 2], padding='SAME', scope='pooling_py1')

        # pyramid atrous large kernel module
        py1_alk = atrous_large_kernel(py1, 256, 15, 1, 'alk_py1')  # 16x16
        py2_alk = atrous_large_kernel(py2, 256, 15, 2, 'alk_py2')  # 32x32
        py4_alk = atrous_large_kernel(py4, 256, 15, 4, 'alk_py4')  # 64x64

        # Fusion
        py2_shape = py2.get_shape().as_list()
        py1_alk = tf.image.resize_bilinear(py1_alk, (py2_shape[1], py2_shape[2]))
        py2_alk = py2_alk + py1_alk

        py4_shape = py4.get_shape().as_list()
        py2_alk = tf.image.resize_bilinear(py2_alk, (py4_shape[1], py4_shape[2]))
        py4_alk = py4_alk + py2_alk

        net_alk = py4_alk
    # Classifier
    img_shape = image.get_shape().as_list()
    fc8s = slim.conv2d(net_alk, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_cls', trainable=is_training,
                       activation_fn=None,normalizer_fn=None,
                       weights_initializer=init_ops.variance_scaling_initializer)
    logits = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])
    label_pred = tf.expand_dims(tf.argmax(logits, axis=3, name="prediction"), dim=3)

    return label_pred, logits

def inference_Pyramid_ALKNet2(image, is_training):

    # Encoder
    net, end_points = resnet_backbone(image)
    # Decoder
    with tf.variable_scope("decoder"):
        # down dimension and build pyramid scales
        py4 = slim.conv2d(net, 512, [1, 1], scope='py4_dim_down', trainable=is_training,
                         weights_initializer=init_ops.variance_scaling_initializer)
        py2 = slim.conv2d(py4, 512, [3, 3], scope='pooling_py2', stride=2,
                         weights_initializer=init_ops.variance_scaling_initializer)  # 32x32

        py1 = slim.conv2d(py2, 512, [3, 3], scope='pooling_py1', stride=2,
                         weights_initializer=init_ops.variance_scaling_initializer)  # 16x16

        # pyramid atrous large kernel module
        py1_alk = atrous_large_kernel(py1, 256, 15, 1, 'alk_py1')  # 16x16
        py2_alk = atrous_large_kernel(py2, 256, 15, 2, 'alk_py2')  # 32x32
        py4_alk = atrous_large_kernel(py4, 256, 15, 4, 'alk_py4')  # 64x64

        # Fusion
        py2_shape = py2.get_shape().as_list()
        py1_alk = tf.image.resize_bilinear(py1_alk, (py2_shape[1], py2_shape[2]))
        py2_alk = py2_alk + py1_alk

        py4_shape = py4.get_shape().as_list()
        py2_alk = tf.image.resize_bilinear(py2_alk, (py4_shape[1], py4_shape[2]))
        py4_alk = py4_alk + py2_alk

        net_alk = py4_alk
    # Classifier
    img_shape = image.get_shape().as_list()
    fc8s = slim.conv2d(net_alk, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_cls', trainable=is_training,
                       activation_fn=None,normalizer_fn=None,
                       weights_initializer=init_ops.variance_scaling_initializer)
    logits = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])
    label_pred = tf.expand_dims(tf.argmax(logits, axis=3, name="prediction"), dim=3)

    return label_pred, logits

def inference_py_alk_512(image, is_training):

    net, end_points = resnet_backbone(image)

    def large_kernel2_he(x, c, k, r, name):
        '''
        large kernel for facade
        :param x:  input feature
        :param c: output channel
        :param k: kernel size
        :param r: rate for conv
        :return:
        '''
        # 1xk + kx1
        left_1 = slim.conv2d(x, c, [1, k], scope=name + '/left_1', rate=r,
                             weights_initializer=init_ops.variance_scaling_initializer)
        left = slim.conv2d(left_1, c, [k, 1], scope=name + '/left_2', rate=r,
                           weights_initializer=init_ops.variance_scaling_initializer)

        right_1 = slim.conv2d(x, c, [k, 1], scope=name + '/right_1', rate=r,
                              weights_initializer=init_ops.variance_scaling_initializer)
        right = slim.conv2d(right_1, c, [1, k], scope=name + '/right_2', rate=r,
                            weights_initializer=init_ops.variance_scaling_initializer)

        y = left + right

        return y

    with tf.variable_scope("decoder"):
        # py
        a3 = slim.conv2d(net, 512, [1, 1], scope='a3_dim_down', trainable=is_training,
                         weights_initializer=init_ops.variance_scaling_initializer)

        a2 = slim.max_pool2d(a3, [2, 2], padding='SAME', scope='poola1')
        a1 = slim.max_pool2d(a2, [2, 2], padding='SAME', scope='poola2')

        # a2 = slim.conv2d(a3, 512, [3, 3], scope='poola1', stride=2,
        #                  weights_initializer=init_ops.variance_scaling_initializer)  # 32x32
        #
        # a1 = slim.conv2d(a2, 512, [3, 3], scope='poola2', stride=2,
        #                  weights_initializer=init_ops.variance_scaling_initializer)  # 16x16

        a1 = large_kernel2_he(a1, 256, 15, 1, 'lk_a1')  # 16x16
        a2 = large_kernel2_he(a2, 256, 15, 2, 'lk_a2')  # 32x32
        a3 = large_kernel2_he(a3, 256, 15, 4, 'lk_a3')  # 64x64

        a2_shape = a2.get_shape().as_list()
        net_shape = net.get_shape().as_list()

        # # image pooling
        # img_pooling = tf.reduce_mean(net, [1, 2], name='image_level_global_pooling', keep_dims=True)
        # img_pooling = slim.conv2d(img_pooling, 256, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
        # img_pooling = tf.image.resize_bilinear(img_pooling, (net_shape[1], net_shape[2]))
        #
        # Attention
        a1 = tf.image.resize_bilinear(a1, (a2_shape[1], a2_shape[2]))
        a2 = a2 + a1
        a2 = tf.image.resize_bilinear(a2, (net_shape[1], net_shape[2]))
        a3 = a3 + a2

        net = a3

    img_shape = image.get_shape().as_list()
    fc8s = slim.conv2d(net, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits1', trainable=is_training,
                       activation_fn=None,normalizer_fn=None,
                       weights_initializer=init_ops.variance_scaling_initializer)
    output_seg_8s = tf.image.resize_images(fc8s, [img_shape[1], img_shape[2]])
    label_pred = tf.expand_dims(tf.argmax(output_seg_8s, axis=3, name="prediction"), dim=3)

    return label_pred, output_seg_8s