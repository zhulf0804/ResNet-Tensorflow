#coding=utf-8

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import math
import numpy as np


# ResNet
Layers_50 = [3, 4, 6, 3]
Layers_101 = [3, 4, 23, 3]
CHANNELS = 3
INIT_FEATURES = 64
CLASSES = 100

#Used for BN
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def weight_variable(shape, stddev=None, name='weight'):
    if stddev == None:
        if len(shape) == 4:
            stddev = math.sqrt(2. / (shape[0] * shape[1] * shape[2]))
        else:
            stddev = math.sqrt(2. / shape[0])
    else:
        stddev = 0.1
    initial = tf.truncated_normal(shape, stddev=stddev)
    W = tf.Variable(initial, name=name)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
    return W


def bias_variable(shape, name='bias'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def bn_layer(x, is_training):
    output = tf.contrib.layers.batch_norm(x, scale=True, is_training=is_training, updates_collections=None)
    return output

def batch_norm(inputs, training):

  return tf.layers.batch_normalization(
      inputs=inputs, axis=-1,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)

def projection(input, output_features, is_training, stride):

    input_shape = input.get_shape().as_list()
    input_features = input_shape[-1]
    weight_1 = weight_variable([1, 1, input_features, output_features])
    input = tf.nn.conv2d(input, weight_1, [1, stride, stride, 1], padding='SAME')
    input = batch_norm(input, is_training)

    return input

def residual_bottleneck_block(input, compressed_features, is_training, Projection, Stride=2):

    shortcut = input

    input_shape = input.get_shape().as_list()
    input_features = input_shape[-1]
    output_features = 4 * compressed_features

    if Projection:
        shortcut = projection(input, output_features, is_training, Stride)


    weight_1_0 = weight_variable([1, 1, input_features, compressed_features])
    weight_3 = weight_variable([3, 3, compressed_features, compressed_features])
    weight_1_1 = weight_variable([1, 1, compressed_features, output_features])

    input = tf.nn.conv2d(input, weight_1_0, [1, 1, 1, 1], padding='SAME')
    input = batch_norm(input, is_training)
    input = tf.nn.relu(input)

    if Stride == 2:
        input = tf.nn.conv2d(input, weight_3, [1, 2, 2, 1], padding='SAME')
    else:
        input = tf.nn.conv2d(input, weight_3, [1, 1, 1, 1], padding='SAME')

    input = batch_norm(input, is_training)
    input = tf.nn.relu(input)

    input = tf.nn.conv2d(input, weight_1_1, [1, 1, 1, 1], padding='SAME')
    input = batch_norm(input, is_training)

    input += shortcut

    input = tf.nn.relu(input)

    return input


def block_layer(input, layers, compressed_features, is_training, i):

    with tf.name_scope("conv%d_0" %i):
        if i == 2:
            input = residual_bottleneck_block(input, compressed_features, is_training, Projection=True, Stride=1)
        else:
            input = residual_bottleneck_block(input, compressed_features, is_training, Projection=True)

    for j in range(1, layers):
        input = residual_bottleneck_block(input, compressed_features, is_training, Projection=False, Stride=1)

    return input


def resnet(input, is_training, keep_prob, layers=101):
    if layers == 101:
        Layers = Layers_101
    elif layers == 50:
        Layers = Layers_50
    else:
        raise Exception("layers %f not support" %layers)

    with tf.name_scope("conv1"):
        weight_7 = weight_variable([7, 7, CHANNELS, INIT_FEATURES])
        input = tf.nn.conv2d(input, weight_7, [1, 2, 2, 1], padding='SAME')

        input = batch_norm(input, is_training)
        input = tf.nn.relu(input)

        input = tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    for i in range(2, len(Layers) + 2):
        with tf.name_scope("conv%d_x" % i):
            compressed_features = INIT_FEATURES * pow(2, i-2)
            print(compressed_features)
            layers = Layers[i-2]

            input = block_layer(input, layers, compressed_features, is_training, i)


    with tf.name_scope("average_pool"):
        input = tf.reduce_mean(input_tensor=input, axis=[1, 2], keepdims=True)
        input = tf.squeeze(input)

    with tf.name_scope("classification"):
        input_shape = input.get_shape().as_list()

        weight_fc = weight_variable(shape=[input_shape[-1], CLASSES])
        input = tf.matmul(input, weight_fc)

    return input


if __name__ == '__main__':
    x = tf.constant(shape=[4, 224, 224, 3], value=1.0)
    x = resnet(x, True, keep_prob=1.0)
    print(x)
