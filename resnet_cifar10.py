#coding=utf-8

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import math
import numpy as np


#Used for BN
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

# for cifar-10
CLASSES = 10
CHANNELS = 3

cifar_INIT_FEATURES = 16
cifar_Layers_cifar = []
cifar_ResNet_size = 56
cifar_ResNet_block_num = 3
cifar_block_size = (cifar_ResNet_size - 2) // 6  #9
cifar_stride = [1, 2, 2]



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
    #tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
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


def residual_block(input, output_features, is_training, i, j):

    input_shape = input.get_shape().as_list()
    input_features = input_shape[-1]

    shortcut = input

    if j == 0:
        shortcut = projection(input, output_features, is_training, stride=cifar_stride[i - 2])


    input = batch_norm(input, is_training)
    input = tf.nn.relu(input)

    weight_3_0 = weight_variable([3, 3, input_features, output_features])

    if j == 0:
        input = tf.nn.conv2d(input, weight_3_0, [1, cifar_stride[i - 2], cifar_stride[i - 2], 1], padding='SAME')
    else:
        input = tf.nn.conv2d(input, weight_3_0, [1, 1, 1, 1], padding='SAME')


    input = batch_norm(input, is_training)
    input = tf.nn.relu(input)

    weight_3_1 = weight_variable([3, 3, output_features, output_features])
    input = tf.nn.conv2d(input, weight_3_1, [1, 1, 1, 1], padding='SAME')

    input += shortcut

    return input

def block_layer(input, layers, output_features, is_training, i):


    for j in range(0, layers):
        with tf.name_scope("conv%d_%d" %(i, j)):
            input = residual_block(input, output_features, is_training, i, j)

    return input


def resnet_cifar_10(input, is_training, keep_prob):


    with tf.name_scope("conv1"):
        weight_3 = weight_variable([3, 3, CHANNELS, cifar_INIT_FEATURES])
        input = tf.nn.conv2d(input, weight_3, [1, 1, 1, 1], padding='SAME')

        #input = batch_norm(input, is_training)
        #input = tf.nn.relu(input)


    for i in range(2, cifar_ResNet_block_num + 2):
        with tf.name_scope("conv%d_x" % i):
            output_features = cifar_INIT_FEATURES * pow(2, i-2)
            #print(output_features)

            input = block_layer(input, cifar_block_size, output_features, is_training, i)

    input = batch_norm(input, is_training)
    input = tf.nn.relu(input)

    with tf.name_scope("average_pool"):
        input = tf.reduce_mean(input_tensor=input, axis=[1, 2], keepdims=True)
        input = tf.squeeze(input)

    with tf.name_scope("classification"):
        input_shape = input.get_shape().as_list()

        weight_fc = weight_variable(shape=[input_shape[-1], CLASSES])
        bias = bias_variable([CLASSES])
        input = tf.matmul(input, weight_fc) + bias

    return input



if __name__ == '__main__':
    x = tf.constant(shape=[4, 224, 224, 3], value=1.0)
    x = resnet_cifar_10(x, True, keep_prob=1.0)

    for v in tf.trainable_variables():
        print(v)
    print(x)
