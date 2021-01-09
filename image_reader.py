import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import tensorflow as tf
import config as cfg
import cv2
from scipy import misc
import random

def image_resizing(img, label):
    '''
    Random resize the images and labels between 0.5 to 1.5 for height or width
    :param img:
    :param label:
    :return: img and label
    '''

    scale = tf.cast(np.random.uniform(0.75, 1.25), dtype=tf.float32)
    img_h = tf.shape(img)[0]
    img_w = tf.shape(img)[1]
    h_scale = tf.to_int32(tf.to_float(img_h) * scale)
    w_scale = tf.to_int32(tf.to_float(img_w) * scale)

    if np.random.uniform(0, 1) < 0.5:
        h_new = h_scale
        w_new = img_w
    else:
        h_new = img_h
        w_new = w_scale

    new_shape = tf.stack([h_new, w_new])
    img_d = tf.image.resize_images(img, new_shape)
    label_d = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label_d = tf.squeeze(label_d, squeeze_dims=[0])

    return img_d, label_d

def image_scaling(img, label):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
      mask: 3 layer(top, mid, bot) mask to scale.
      boundary: boundary mask to scale.
      scale: ECP: [0.38, 1.0] [0.58, 1.25] [0.75, 1.5]
             eTRIMS:[0.33, 0.75] [0.5, 1.0] [0.66, 1.25]
    """

    # # fixed scales: no useless because the scale is fixed at one value
    # scales = [0.75, 0.87, 1.0, 1.15, 1.3, 1.45, 1.6, 1.75]
    # sc = random.sample(scales, 1)
    # print(sc)
    # scale = tf.convert_to_tensor(sc, dtype=tf.float32)

    # random scales range(0.75, 1.75)
    scale = tf.random_uniform([1], minval=cfg.minScale, maxval=cfg.maxScale, dtype=tf.float32, seed=None)

    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])

    return img, label


def image_mirroring(img, label):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
      mask: 3 layer mask to mirror.
      boundary: boundary mask to mirror.
    """

    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)

    return img, label


def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      mask: 3 layer mask to crop/pad.
      boundary: boundary mask to crop/pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label  # Needs to be subtracted and later added due to 0 padding.

    combined = tf.concat(axis=2, values=[image, label])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]),
                                                tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]

    combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, 4])
    img_crop = combined_crop[:, :, :last_image_dim]

    label_crop = combined_crop[:, :, last_image_dim:last_image_dim + 1]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)


    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h, crop_w, 1))


    return img_crop, label_crop

def get_image_and_labels(image, label, crop_h, crop_w):
    # Set static shape so that tensorflow knows shape at compile time.

    # # For other 512 x 512
    # new_shape = tf.squeeze(tf.stack([crop_h, crop_w]))
    # image = tf.image.resize_images(image, new_shape)
    # label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    # label = tf.squeeze(label, squeeze_dims=[0])

    image.set_shape((crop_h, crop_w, 3))
    label.set_shape((crop_h, crop_w, 1))

    return image, label


def random_brightness_contrast_hue_satu(img):
    '''
    Random birght and countrast
    :param img:
    :return:
    '''
    if np.random.uniform(0, 1) < 0.5:
        distorted_image = tf.image.random_brightness(img, max_delta=32./255.)
        distorted_image = tf.image.random_saturation(distorted_image, lower=0.5, upper=1.5)
        distorted_image = tf.image.random_hue(distorted_image, max_delta=0.2)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.5, upper=1.5)
    else:
        distorted_image = tf.image.random_brightness(img, max_delta=32./255.)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.5, upper=1.5)
        distorted_image = tf.image.random_saturation(distorted_image, lower=0.5, upper=1.5)
        distorted_image = tf.image.random_hue(distorted_image, max_delta=0.2)

    image = distorted_image

    return image



def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/label
                                                        /path/to/mask  /path/to/boundary '.

    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    labels = []

    for line in f:
        try:
            image, label  = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            image = label = line.strip("\n")
        images.append(data_dir + image)
        labels.append(data_dir + label)


    return images, labels


def read_images_from_disk(input_queue, input_size, random_scale, random_resize, random_mirror, random_color, random_crop_pad,
                          ignore_label, img_mean):  # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.

    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      random_color: random brightness, contrast, hue and saturation.
      random_crop_pad: random crop and padding for h and w of image
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.

    Returns:
      Two tensors: the decoded image and its mask.
    """

    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])


    img = tf.image.decode_jpeg(img_contents, channels=3)
    img = tf.cast(img, dtype=tf.float32)
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)      # B G R


    label = tf.image.decode_png(label_contents, channels=1)


    if input_size is not None:
        h, w = input_size

        # Randomly scale the images and labels.
        if random_scale:
            img, label = image_scaling(img, label)

        if random_resize:
            img, label = image_resizing(img, label)

        # Randomly mirror the images and labels.
        if random_mirror:
            img, label = image_mirroring(img, label)

        # random_brightness_contrast_hue_satu
        if random_color:
            img = random_brightness_contrast_hue_satu(img)

        # Randomly crops the images and labels.
        if random_crop_pad:
            img, label  = random_crop_and_pad_image_and_labels(img, label, h, w, ignore_label)
        else:
            img, label = get_image_and_labels(img, label, h, w)

    # Extract mean.
    img -= img_mean

    return img, label


class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size,
                 random_scale, random_resize, random_mirror, random_color, random_crop_pad, ignore_label, img_mean, coord):
        '''Initialise an ImageReader.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          random_color: whether to randomly brightness, contrast, hue and satr.
          random_crop_pad: whether to randomly corp and pading images.
          ignore_label: index of label to ignore during the training.
          img_mean: vector of mean colour values.
          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord

        self.image_list, self.label_list = read_labeled_image_list(self.data_dir, self.data_list)

        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)


        self.queue = tf.train.slice_input_producer([self.images, self.labels],
                                                   shuffle=False)  # not shuffling if it is val #
        # True: not equal, False: pre-processing data list. Default: False

        self.image, self.label = read_images_from_disk(self.queue, self.input_size,
                                                                                 random_scale, random_resize, random_mirror,
                                                                                 random_color, random_crop_pad,
                                                                                 ignore_label, img_mean)

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.

        Args:
          num_elements: the batch size.

        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        image_batch, label_batch = tf.train.batch(
            [self.image, self.label],
            num_elements)
        return image_batch, tf.cast(label_batch, dtype=tf.int32)

    def getqueue(self, num_elements):
        '''Pack images and labels into a batch.

        Args:
          num_elements: the batch size.

        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        image_queue = tf.train.batch(
            [self.queue],
            num_elements)
        return image_queue


if __name__ == '__main__':

    input_size = (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
    # # Create queue coordinator.
    coord = tf.train.Coordinator()
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            cfg.train_data_dir,
            cfg.train_data_list,
            input_size,
            cfg.random_scale,
            cfg.random_resize,
            cfg.random_mirror,
            cfg.random_color,
            cfg.random_crop_pad,
            cfg.ignore_label,
            cfg.IMG_MEAN,
            coord)
        image_batch, label_batch = reader.dequeue(cfg.batch_size)
        # ques = reader.getqueue(cfg.batch_size)

    with tf.Session() as se:
        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=se)

        # f = open('queue.txt', 'w')
        # for i in range(40000):
        #     que = se.run(ques)
        #     print(que)
        #     f.write(str(que))
        # f.close()

        imgs, labels = se.run([image_batch, label_batch])
        img = np.array(imgs[0] + cfg.IMG_MEAN)
        label = np.squeeze(labels[0], axis=2) * 20

        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('3_img5.png', img)
        cv2.imwrite('3_label.png', label)

        coord.request_stop()
        coord.join(threads)
    print('Done')