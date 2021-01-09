import tensorflow as tf
import config as cfg

def cross_entropy_loss(logits, annotation):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                         labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                         name="entropy"))
    return loss

def cross_entropy_loss_global_context(logits, annotation):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                         labels=tf.squeeze(annotation, squeeze_dims=[2]),
                                                                         name="entropy"))
    return loss

def l1_loss(logits, labels, L1_lambda=1.0):
    loss = L1_lambda * tf.reduce_mean(tf.abs(logits - labels))

    return loss

def l2_loss(logits, labels):
    loss = tf.square(logits - labels) / 2
    loss = tf.reduce_mean(loss)

    return loss

def cross_entropy_loss2(logits, labels, threshold=1.):
    #ano_squ = tf.squeeze(labels, squeeze_dims=[3])
    ano_onehot = tf.cast(labels, dtype=tf.float32) #tf.cast(tf.one_hot(indices=ano_squ, depth=cfg.NUM_OF_CLASSESS, on_value=1, off_value=0), dtype=tf.float32)

    loss = -tf.reduce_mean(ano_onehot * tf.log(tf.clip_by_value(logits, 1e-8, 1)) +
                           (1 - ano_onehot) * tf.log(tf.clip_by_value(1 - logits, 1e-8, 1)))

    return loss


def OHEM_cross_entropy_loss(logits, labels, threshold=0.5):
    ano_squ = tf.squeeze(labels, squeeze_dims=[3])
    ano_onehot = tf.cast(tf.one_hot(indices=ano_squ, depth=cfg.NUM_OF_CLASSESS, on_value=1, off_value=0), dtype=tf.float32)

    max_pro = tf.reduce_max(ano_onehot * tf.clip_by_value(tf.nn.softmax(logits), 1e-8, 1), axis=3)
    zero = tf.zeros(max_pro.get_shape().as_list())
    one = tf.ones(max_pro.get_shape().as_list())
    num = tf.cast(tf.reduce_sum(tf.where(max_pro <= threshold, one, zero)), dtype=tf.float32)
    ohem = tf.expand_dims(tf.where(max_pro <= threshold, one, zero), axis=3)
    total_loss = -tf.reduce_sum(ohem * ano_onehot * tf.log(tf.clip_by_value(tf.nn.softmax(logits), 1e-8, 1)) +
                           ohem * (1 - ano_onehot) * tf.log(tf.clip_by_value(1 - tf.nn.softmax(logits), 1e-8, 1)))
    loss = total_loss / num
    return loss, num, total_loss



def smooth_L1_loss(logits, labels):
    x = tf.subtract(logits, labels)
    loss = tf.reduce_mean(tf.where(tf.less_equal(tf.abs(x), 1.0),
                                   tf.multiply(0.5, tf.pow(x, 2.0)),
                                   tf.subtract(tf.abs(x), 0.5)))
    return loss

def L1_loss(logits, labels):
    x = tf.subtract(logits, labels)
    loss = tf.reduce_mean(tf.abs(x))
    return loss


def weighted_cross_entropy_loss(decode_logits, binary_label):
    decode_logits_reshape = tf.reshape(
        decode_logits,
        shape=[decode_logits.get_shape().as_list()[0],
               decode_logits.get_shape().as_list()[1] * decode_logits.get_shape().as_list()[2],
               decode_logits.get_shape().as_list()[3]])

    binary_label_reshape = tf.reshape(
        binary_label,
        shape=[binary_label.get_shape().as_list()[0],
               binary_label.get_shape().as_list()[1] * binary_label.get_shape().as_list()[2]])
    binary_label_reshape = tf.one_hot(binary_label_reshape, depth=cfg.NUM_OF_CLASSESS)
    class_weights = [0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    weights_loss = tf.reduce_sum(tf.multiply(binary_label_reshape, class_weights), 2)
    binary_segmentation_loss = tf.losses.softmax_cross_entropy(onehot_labels=binary_label_reshape,
                                                               logits=decode_logits_reshape,
                                                               weights=weights_loss)
    binary_segmentation_loss = tf.reduce_mean(binary_segmentation_loss)

    return binary_segmentation_loss

def weighted_cross_entropy_loss_4class(decode_logits, binary_label):
    decode_logits_reshape = tf.reshape(
        decode_logits,
        shape=[decode_logits.get_shape().as_list()[0],
               decode_logits.get_shape().as_list()[1] * decode_logits.get_shape().as_list()[2],
               decode_logits.get_shape().as_list()[3]])

    binary_label_reshape = tf.reshape(
        binary_label,
        shape=[binary_label.get_shape().as_list()[0],
               binary_label.get_shape().as_list()[1] * binary_label.get_shape().as_list()[2]])
    binary_label_reshape = tf.one_hot(binary_label_reshape, depth=cfg.NUM_OF_CLASSESS)
    class_weights = [0, 1.0, 1.0, 1.0, 1.0]
    weights_loss = tf.reduce_sum(tf.multiply(binary_label_reshape, class_weights), 2)
    binary_segmentation_loss = tf.losses.softmax_cross_entropy(onehot_labels=binary_label_reshape,
                                                               logits=decode_logits_reshape,
                                                               weights=weights_loss)
    binary_segmentation_loss = tf.reduce_mean(binary_segmentation_loss)

    return binary_segmentation_loss


def boundary_loss(decode_logits, binary_label):
    decode_logits_reshape = tf.reshape(
        decode_logits,
        shape=[decode_logits.get_shape().as_list()[0],
               decode_logits.get_shape().as_list()[1] * decode_logits.get_shape().as_list()[2],
               decode_logits.get_shape().as_list()[3]])

    binary_label_reshape = tf.reshape(
        binary_label,
        shape=[binary_label.get_shape().as_list()[0],
               binary_label.get_shape().as_list()[1] * binary_label.get_shape().as_list()[2]])
    binary_label_reshape = tf.one_hot(binary_label_reshape, depth=2)

    class_weights = tf.constant([[1., 12.]])  # exclude outlier
    weights_loss = tf.reduce_sum(tf.multiply(binary_label_reshape, class_weights), 2)
    binary_segmentation_loss = tf.losses.softmax_cross_entropy(onehot_labels=binary_label_reshape,
                                                               logits=decode_logits_reshape,
                                                               weights=weights_loss)
    binary_segmentation_loss = tf.reduce_mean(binary_segmentation_loss)

    return binary_segmentation_loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def lovasz_softmax_flat(probas, labels, classes='all'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = probas.shape[1]
    losses = []
    present = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = tf.cast(tf.equal(labels, c), probas.dtype)  # foreground for class c
        if classes == 'present':
            present.append(tf.reduce_sum(fg) > 0)
        errors = tf.abs(fg - probas[:, c])
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                      )
    if len(class_to_sum) == 1:  # short-circuit mean when only one class
        return losses[0]
    losses_tensor = tf.stack(losses)
    if classes == 'present':
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    loss = tf.reduce_mean(losses_tensor)
    return loss
def flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels

def lovasz_softmax(probas, labels, classes='all', per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    if per_image:
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore, order)
            return lovasz_softmax_flat(prob, lab, classes=classes)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore, order), classes=classes)
    return loss

def IoU_boundary_loss(logits, labels):
    ano_squ = tf.squeeze(labels, squeeze_dims=[3])
    ano_onehot = tf.cast(tf.one_hot(indices=ano_squ, depth=cfg.NUM_OF_CLASSESS, on_value=1, off_value=0),
                         dtype=tf.float32)
    logits = tf.nn.softmax(logits)

    n = tf.reduce_sum(logits * ano_onehot)
    d = tf.reduce_sum(logits + ano_onehot - logits * ano_onehot)
    l_iou = 1 - (n/d)

    return l_iou

# def _get_loss(logits, label):
#     import numpy as np
#     import cv2
#     total_loss = 0
#     logits = np.array(logits, np.float32)
#     is_print = False
#     if is_print:
#         print('label: ', label.shape)
#         print('logits: ', logits.shape)
#
#     for b in range(cfg.batch_size):
#         im = label[b]
#         im_copy = im.copy()
#         im_copy[im != 1] = 0
#         im_copy = np.array(im_copy, np.uint8)
#         if is_print:
#             print('imcopy', im_copy.shape)
#         _, cnts, _ = cv2.findContours(im_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         x = []
#         w = []
#         y = []
#         h = []
#         for cors in cnts:
#             cors = np.squeeze(cors, 1)
#             if max(cors[:, 0]) - min(cors[:, 0]) > 5 and max(cors[:, 1]) - min(cors[:, 1]) > 5:
#                 x.append(min(cors[:, 0]))
#                 w.append(max(cors[:, 0]) - min(cors[:, 0]))
#                 y.append(min(cors[:, 1]))
#                 h.append(max(cors[:, 1]) - min(cors[:, 1]))
#
#         x = np.array(x)[:, np.newaxis]
#         w = np.array(w)[:, np.newaxis]
#         y = np.array(y)[:, np.newaxis]
#         h = np.array(h)[:, np.newaxis]
#         # Define the max box number
#         maxnum = 55
#
#         cors = np.concatenate([y, x, h, w], axis=1)
#         cors = np.array(cors, np.int32)
#         number = len(cors)
#         if is_print:
#             print('Cors:', cors)
#
#         # # shuffle boxes
#         # np.random.shuffle(cors)
#         # number = len(cors)
#         # # fill the max box num
#         # n = int(np.ceil(maxnum / number) - 1)
#         # cor_list = cors
#         # for i in range(n):
#         #     cor_list = np.concatenate([cor_list, cors], axis=0)
#         # all_cors = cor_list[0:maxnum, :]
#
#         loss_per_im = 0
#         for n in range(number):
#
#             lab = logits[b, cors[n][0]:cors[n][0]+cors[n][2], cors[n][1]:cors[n][1]+cors[n][3], :]
#             pred = np.argmax(lab, 2)[:, :, np.newaxis]
#             location = np.where(pred == 1)
#
#             if is_print:
#                 print(n)
#                 print('lab:', lab.shape)
#                 print('pred:', pred.shape)
#                 print('location:', location)
#
#
#             min_h = 0
#             max_h = 0
#             min_w = 0
#             max_w = 0
#             if len(location[0]) > 0:
#                 max_h = max(location[0])
#                 min_h = min(location[0])
#                 if is_print:
#                     print('maxh:', min_h, max_h)
#
#             if len(location[1]) > 0:
#                 max_w = max(location[1])
#                 min_w = min(location[1])
#                 if is_print:
#                     print('maxx:', min_w, max_w)
#
#             length = len(location[0])
#
#             # direct Y
#             y_lists = []
#             num_h = 0
#             total_h = 0
#             for h in range(min_h, max_h):
#                 y_h = []
#                 count = 0
#                 for l in range(length):
#                     if location[0][l] == h:
#                         y_h.append(location[1][l])
#                         count += 1
#
#                 if count > 0:
#                     y_lists.append(sum(y_h) / count)
#                     num_h += 1
#
#             if num_h > 0:
#                 total_h = sum(y_lists) / num_h
#
#             loss_h = 0
#             for y_list in y_lists:
#                 loss_h += (y_list - total_h) * (y_list - total_h)
#             if num_h > 0:
#                 loss_h = loss_h / num_h
#                 if is_print:
#                     print('loss_h', loss_h)
#
#             # Direct X
#             x_lists = []
#             num_w = 0
#             total_w = 0
#             for w in range(min_w, max_w):
#                 x_h = []
#                 count = 0
#                 for l in range(length):
#                     if location[1][l] == w:
#                         x_h.append(location[0][l])
#                         count += 1
#                 if count > 0:
#                     x_lists.append(sum(x_h) / count)
#                     num_w += 1
#             if num_w > 0:
#                 total_w = sum(x_lists) / num_w
#             loss_w = 0
#             for x_list in x_lists:
#                 loss_w += (x_list - total_w) * (x_list - total_w)
#             if num_w > 0:
#                 loss_w = loss_w / num_w
#                 if is_print:
#                     print('lossw', loss_w)
#
#             loss = loss_h + loss_w
#             loss_per_im += loss
#         if is_print:
#             print('loss per im: ' + str(n+1) +', '+ str(loss_per_im))
#         total_loss += loss_per_im
#
#     return np.array(total_loss, np.float32)
#
#
#
#
#
#
#
# def get_loss(logits, labels):
#     labels = tf.squeeze(labels, 3)
#     loss = tf.py_func(_get_loss, [logits, labels], tf.float32)
#     loss.set_shape([])
#     return loss

def _get_boxes(label):
    import numpy as np
    import cv2

    is_print = False
    if is_print:
        print('label: ', label.shape)
    all_cors = np.zeros([MAXNUM*cfg.batch_size, 4])
    for b in range(cfg.batch_size):
        im = label[b]
        im_copy = im.copy()
        im_copy[im != 1] = 0
        im_copy = np.array(im_copy, np.uint8)
        if is_print:
            print('imcopy', im_copy.shape)
        _, cnts, _ = cv2.findContours(im_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        y1 = []
        x1 = []
        y2 = []
        x2 = []
        for cors in cnts:
            cors = np.squeeze(cors, 1)
            if max(cors[:, 0]) - min(cors[:, 0]) > 10 and max(cors[:, 1]) - min(cors[:, 1]) > 10:
                y1.append(min(cors[:, 1]))
                x1.append(min(cors[:, 0]))
                y2.append(max(cors[:, 1]))
                x2.append(max(cors[:, 0]))

        y1 = np.array(y1)[:, np.newaxis]
        x1 = np.array(x1)[:, np.newaxis]
        y2 = np.array(y2)[:, np.newaxis]
        x2 = np.array(x2)[:, np.newaxis]



        cors = np.concatenate([y1, x1, y2, x2], axis=1)
        cors = np.array(cors, np.int32)
        number = len(cors)
        if is_print:
            print('Cors:', cors)

        # shuffle boxes
        np.random.shuffle(cors)
        number = len(cors)
        # fill the max box num
        n = int(np.ceil(MAXNUM / number) - 1)
        cor_list = cors
        for i in range(n):
            cor_list = np.concatenate([cor_list, cors], axis=0)
        all_cors[b*MAXNUM: (b+1)*MAXNUM, :] =  cor_list[0:MAXNUM, :]

    return np.array(all_cors, np.float32)


def get_boxes(labels):
    labels = tf.squeeze(labels, 3)
    boxes = tf.py_func(_get_boxes, [labels], tf.float32)
    boxes.set_shape([cfg.batch_size * MAXNUM, 4])
    return boxes

# Define the max box number
MAXNUM = 55
def symmetric_loss(logits, labels):
    '''symmetric loss for windows etc.'''
    boxes = get_boxes(labels)
    import numpy as np
    batch_ids = np.zeros([cfg.batch_size * MAXNUM])
    for i in range(cfg.batch_size):
        batch_ids[i*MAXNUM: (i+1)*MAXNUM] = i
    batch_ids = tf.convert_to_tensor(batch_ids, tf.int32)
    pre_pool_size = 28 * 2
    crops = tf.image.crop_and_resize(logits, boxes, batch_ids,
                                     [pre_pool_size, pre_pool_size], name='crops')

    crops_argmax = tf.argmax(crops, axis=3)

    zero = tf.zeros(crops_argmax.shape, tf.float32)
    one = tf.ones(crops_argmax.shape, tf.float32)
    location = tf.where(tf.equal(crops_argmax, 1), one, zero)

    rangex = tf.range(1, pre_pool_size+1, dtype=tf.float32)
    range_x = one * rangex
    range_y = tf.transpose(range_x, [0, 2, 1])

    loca_x = location * range_x
    loca_y = location * range_y


    # Loss direct X
    var_x = tf.reduce_mean(loca_x, axis=2)
    var_x_ = tf.expand_dims(tf.reduce_mean(var_x, axis=1), axis=1)
    loss_x = tf.reduce_sum(tf.abs(var_x - var_x_))

    # Loss direct Y
    var_y = tf.reduce_mean(loca_y, axis=1)
    var_y_ = tf.expand_dims(tf.reduce_mean(var_y, axis=1), axis=1)
    loss_y = tf.reduce_sum(tf.abs(var_y - var_y_))


    return loss_x + loss_y

def shape_loss(logits, labels):
    boxes = get_boxes(labels)
    import numpy as np
    batch_ids = np.zeros([cfg.batch_size * MAXNUM])
    for i in range(cfg.batch_size):
        batch_ids[i * MAXNUM: (i + 1) * MAXNUM] = i
    batch_ids = tf.convert_to_tensor(batch_ids, tf.int32)
    pre_pool_size = 28 * 2
    crops = tf.image.crop_and_resize(logits, boxes, batch_ids,
                                     [pre_pool_size, pre_pool_size], name='crops')

    crops_argmax = tf.argmax(crops, axis=3)

    zero = tf.zeros(crops_argmax.shape, tf.float32)
    one = tf.ones(crops_argmax.shape, tf.float32)
    location = tf.where(tf.equal(crops_argmax, 1), one, zero)

    rangex = tf.range(1, pre_pool_size + 1, dtype=tf.float32)
    range_x = one * rangex
    range_y = tf.transpose(range_x, [0, 2, 1])

    # location of direct X and Y
    loca_x = location * range_x
    loca_y = location * range_y

    # rect cors
    x1 = tf.reduce_min(loca_x, 2)
    x2 = tf.reduce_max(loca_x, 2)
    y1 = tf.reduce_min(loca_y, 1)
    y2 = tf.reduce_max(loca_y, 1)


    # y1 direct
    var_y1 = tf.reduce_min(loca_y, axis=1)
    loss_y1 = tf.reduce_sum(tf.abs(var_y1 - y1))

    # y2 direct
    var_y2 = tf.reduce_max(loca_y, axis=1)
    loss_y2 = tf.reduce_sum(tf.abs(var_y2 - y2))

    # x1 direct
    var_x1 = tf.reduce_min(loca_x, axis=2)
    loss_x1 = tf.reduce_sum(tf.abs(var_x1 - x1))

    # x2 direct
    var_x2 = tf.reduce_max(loca_x, axis=2)
    loss_x2 = tf.reduce_sum(tf.abs(var_x2 - x2))

    return loss_y1 + loss_y2 + loss_x1 + loss_x2


def center_pull_loss(logits, labels, t_pull=0.0):
    '''pull embeding feature to the center'''

    batch, h, w, c = logits.get_shape().as_list()
    pull_loss_total = tf.cast(0.0, tf.float32)
    for b in range(batch):

        ano_squ = tf.squeeze(labels[b], squeeze_dims=[2])
        labels_onehot = tf.cast(tf.one_hot(indices=ano_squ, depth=cfg.NUM_OF_CLASSESS, on_value=1, off_value=0),
                                dtype=tf.int32)

        embeddings = []
        for i in range(1, cfg.NUM_OF_CLASSESS):
            f_mask = tf.cast(labels_onehot[:, :, i], dtype=bool)
            feature = tf.boolean_mask(logits[b], f_mask)
            embeddings.append(feature)

        centers = []
        for feature in embeddings:
            center = tf.expand_dims(tf.reduce_mean(feature, axis=0), axis=0)  # b, c
            centers.append(center)

        # pull loss within a same class
        pull_loss = tf.cast(0.0, tf.float32)
        for feature, center in zip(embeddings, centers):
            dis = tf.norm(feature - center, 2, axis=1) - t_pull
            # dis = tf.nn.relu(dis)
            # dis = ((feature - center) * (feature - center)) / 2
            # dis = tf.abs(feature - center)
            pull_loss += tf.reduce_mean(dis)
        pull_loss = pull_loss / (cfg.NUM_OF_CLASSESS - 1)
        pull_loss_total += pull_loss

    loss = pull_loss_total / batch

    return loss



epsilon = 1e-5
smooth = 1




















