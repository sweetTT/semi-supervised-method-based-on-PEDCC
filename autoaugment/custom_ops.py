# coding=utf-8

"""Contains convenience wrappers for typical Neural Network TensorFlow layers.

Ops that have different behavior during training or eval have an is_training
parameter.

Copied from AutoAugment: https://github.com/tensorflow/models/blob/master/research/autoaugment/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
from absl import flags


arg_scope = tf.contrib.framework.arg_scope
FLAGS = tf.flags.FLAGS


def variable(name, shape, dtype, initializer, trainable):
  """Returns a TF variable with the passed in specifications."""
  var = tf.get_variable(
      name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      trainable=trainable)
  return var


def global_avg_pool(x, scope=None):
  """Average pools away spatial height and width dimension of 4D tensor."""
  assert x.get_shape().ndims == 4
  with tf.name_scope(scope, 'global_avg_pool', [x]):
    kernel_size = (1, int(x.shape[1]), int(x.shape[2]), 1)
    squeeze_dims = (1, 2)
    result = tf.nn.avg_pool(
        x,
        ksize=kernel_size,
        strides=(1, 1, 1, 1),
        padding='VALID',
        data_format='NHWC')
    return tf.squeeze(result, squeeze_dims)


def zero_pad(inputs, in_filter, out_filter):
  """Zero pads `input` tensor to have `out_filter` number of filters."""
  outputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 0],
                            [(out_filter - in_filter) // 2,
                             (out_filter - in_filter) // 2]])
  return outputs


@tf.contrib.framework.add_arg_scope
def batch_norm(inputs,
               update_stats=True,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               is_training=True,
               reuse=None,
               scope=None,
              ):
  """Small wrapper around tf.contrib.layers.batch_norm."""
  batch_norm_op = tf.layers.batch_normalization(
      inputs,
      axis=-1,
      momentum=decay,
      epsilon=epsilon,
      center=center,
      scale=scale,
      training=is_training,
      fused=True,
      trainable=True,
  )
  return batch_norm_op


def stride_arr(stride_h, stride_w):
  return [1, stride_h, stride_w, 1]


@tf.contrib.framework.add_arg_scope
def conv2d(inputs,
           num_filters_out,
           kernel_size,
           stride=1,
           scope=None,
           reuse=None):
  """Adds a 2D convolution.

  conv2d creates a variable called 'weights', representing the convolutional
  kernel, that is convolved with the input.

  Args:
    inputs: a 4D tensor in NHWC format.
    num_filters_out: the number of output filters.
    kernel_size: an int specifying the kernel height and width size.
    stride: an int specifying the height and width stride.
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and its variables should be reused.
  Returns:
    a tensor that is the result of a convolution being applied to `inputs`.
  """
  with tf.variable_scope(scope, 'Conv', [inputs], reuse=reuse):
    num_filters_in = int(inputs.shape[3])
    weights_shape = [kernel_size, kernel_size, num_filters_in, num_filters_out]

    # Initialization
    n = int(weights_shape[0] * weights_shape[1] * weights_shape[3])
    weights_initializer = tf.random_normal_initializer(
        stddev=np.sqrt(2.0 / n))

    weights = variable(
        name='weights',
        shape=weights_shape,
        dtype=tf.float32,
        initializer=weights_initializer,
        trainable=True)
    strides = stride_arr(stride, stride)
    outputs = tf.nn.conv2d(
        inputs, weights, strides, padding='SAME', data_format='NHWC')
    return outputs


@tf.contrib.framework.add_arg_scope
def fc(inputs,
       feature_dim,
       scope=None,
       reuse=None):
  """Creates a fully connected layer applied to `inputs`.

  Args:
    inputs: a tensor that the fully connected layer will be applied to. It
      will be reshaped if it is not 2D.
    num_units_out: the number of output units in the layer.  ##类别数
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and its variables should be reused.

  Returns:
     a tensor that is the result of applying a linear matrix to `inputs`.
  """
  if len(inputs.shape) > 2:
    inputs = tf.reshape(inputs, [int(inputs.shape[0]), -1])

  with tf.variable_scope(scope, 'FC', [inputs], reuse=reuse):
    num_units_in = inputs.shape[1]
    weights_shape = [num_units_in, feature_dim]
    unif_init_range = 1.0 / (feature_dim)**(0.5)   ##1/根号10
    weights_initializer = tf.random_uniform_initializer(
        -unif_init_range, unif_init_range)   ###初始化均匀分布  ？？？
    weights = variable(
        name='weights',
        shape=weights_shape,
        dtype=tf.float32,
        initializer=weights_initializer,
        trainable=True)
    bias_initializer = tf.constant_initializer(0.0)  ###常量初始化，初始化为0
    biases = variable(
        name='biases',
        shape=[feature_dim,],
        dtype=tf.float32,
        initializer=bias_initializer,
        trainable=True)
    outputs = tf.nn.xw_plus_b(inputs, weights, biases)
    return outputs


@tf.contrib.framework.add_arg_scope
def avg_pool(inputs, kernel_size, stride=2, padding='VALID', scope=None):
  """Wrapper around tf.nn.avg_pool."""
  with tf.name_scope(scope, 'AvgPool', [inputs]):
    kernel = stride_arr(kernel_size, kernel_size)
    strides = stride_arr(stride, stride)
    return tf.nn.avg_pool(
        inputs,
        ksize=kernel,
        strides=strides,
        padding=padding,
        data_format='NHWC')


# consieLinear层 实现了norm的fea与norm weight的点积计算，服务于margin based softmax loss
# 将w替换成pedcc，固定
# class CosineLinear_PEDCC(nn.Module):
#     def __init__(self, in_features, out_features, is_pedcc):
#         super(CosineLinear_PEDCC, self).__init__()
#
#         self.in_features = in_features
#         self.out_features = out_features
#         if is_pedcc:
#             self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=False)
#             #self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
#             map_dict = read_pkl()
#             tensor_empty = torch.Tensor([]).cuda()
#             for label_index in range(self.out_features):
#                 tensor_empty = torch.cat((tensor_empty, map_dict[label_index].float().cuda()), 0)
#             label_40D_tensor = tensor_empty.view(-1, self.in_features).permute(1, 0)
#             label_40D_tensor = label_40D_tensor.cuda()
#             self.weight.data = label_40D_tensor
#         else:
#             self.weight = Parameter(torch.FloatTensor(out_features, in_features))
#             nn.init.xavier_uniform_(self.weight)
#         #print(self.weight.data)
#
#     def forward(self, input):
#         x = input  # size=(B,F)    F is feature len
#         w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features
#
#         ww = w.renorm(2, 1, 1e-5).mul(1e5)  # weights normed
#         xlen = x.pow(2).sum(1).pow(0.5)  # size=B
#         wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum
#
#         cos_theta = x.mm(ww)  # size=(B,Classnum)  x.dot(ww)
#         cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)  #
#         cos_theta = cos_theta.clamp(-1, 1)
#         cos_theta = cos_theta * xlen.view(-1, 1)
#
#         return cos_theta  # size=(B,Classnum,1)

