from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim

""" Implementation of SEA """
def SEA(net, name = 'SEA'):
  batchsize, height, width, in_channels = net.get_shape().as_list()
  # GAP
  net_GAP = tf.reduce_mean(net, axis=[1,2], keep_dims=True)


  fc1 = slim.conv2d(net_GAP, 16, 1, padding='VALID', activation_fn=tf.nn.relu, normalizer_fn=None, scope=name+'fc1')
  global_conv = slim.conv2d(fc1, in_channels, 1, padding='VALID', activation_fn=tf.nn.sigmoid, normalizer_fn=None, scope=name+'fc2')

  atten_3D = tf.reshape(global_conv, [tf.shape(net_GAP)[0], 1, 1, in_channels])

  scale = net * atten_3D

  return scale
