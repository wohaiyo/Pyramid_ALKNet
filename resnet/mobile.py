import functools
import tensorflow as tf


from tensorflow.contrib.slim.nets import resnet_utils

from resnet.mobilenet import mobilenet_v2

slim = tf.contrib.slim

# Default end point for MobileNetv2.
_MOBILENET_V2_FINAL_ENDPOINT = 'layer_18'

def _mobilenet_v2(net,
                  depth_multiplier,
                  output_stride,
                  reuse=None,
                  scope=None,
                  final_endpoint=None):
  """Auxiliary function to add support for 'reuse' to mobilenet_v2.

  Args:
    net: Input tensor of shape [batch_size, height, width, channels].
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    output_stride: An integer that specifies the requested ratio of input to
      output spatial resolution. If not None, then we invoke atrous convolution
      if necessary to prevent the network from reducing the spatial resolution
      of the activation maps. Allowed values are 8 (accurate fully convolutional
      mode), 16 (fast fully convolutional mode), 32 (classification mode).
    reuse: Reuse model variables.
    scope: Optional variable scope.
    final_endpoint: The endpoint to construct the network up to.

  Returns:
    Features extracted by MobileNetv2.
  """
  with tf.variable_scope(
      scope, 'MobilenetV2', [net], reuse=reuse) as scope:
    return mobilenet_v2.mobilenet_base(
        net,
        conv_defs=mobilenet_v2.V2_DEF,
        depth_multiplier=depth_multiplier,
        min_depth=8 if depth_multiplier == 1.0 else 1,
        divisible_by=8 if depth_multiplier == 1.0 else 1,
        final_endpoint=final_endpoint or _MOBILENET_V2_FINAL_ENDPOINT,
        output_stride=output_stride,
        scope=scope)