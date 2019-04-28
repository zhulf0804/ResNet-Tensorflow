# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import resnet_cifar10 as ResNet
import input_data

BATCH_SIZE = 128
HEIGHT = input_data.HEIGHT
WIDTH = input_data.WIDTH
CHANNELS = 3
CLASSES = ResNet.CLASSES
weight_decay = 1e-4 # L2 normalization

KEEP_PROB = 1.0 #for no use
MAX_STEPS = 80000
initial_lr = 0.1

saved_ckpt_path = './checkpoint/'
saved_summary_train_path = './summary/train/'
saved_summary_test_path = './summary/test/'

with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, HEIGHT, WIDTH, CHANNELS], name='x_input')
    y = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE], name='label')
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    y_onehot = tf.one_hot(y, CLASSES, dtype=tf.float32)

logits = ResNet.resnet_cifar_10(x, True, keep_prob)

with tf.name_scope('regularization'):
    #regularizer = tf.contrib.layers.l2_regularizer(scale)
    #reg_term = tf.contrib.layers.apply_regularization(regularizer)
    l2_loss = weight_decay * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [
            tf.nn.l2_loss(tf.cast(v, tf.float32))
            for v in tf.trainable_variables()
        ])

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits, name='loss'))
    print(loss)
    loss_all = loss + l2_loss
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss_all', loss_all)



with tf.name_scope('learning_rate'):
    lr = tf.Variable(initial_lr, dtype=tf.float32, trainable=False)
    tf.summary.scalar('learning_rate', lr)


#optimizer = tf.train.AdamOptimizer(lr).minimize(loss_all)
optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss_all)
#optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss_all)

with tf.name_scope('accuracy'):
    softmax = tf.nn.softmax(logits, axis=-1)
    correct_prediction = tf.equal(tf.argmax(y_onehot, 1), tf.argmax(softmax, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

train_data = input_data.read_train_data()
test_data = input_data.read_test_data()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # if os.path.exists(saved_ckpt_path):
    ckpt = tf.train.get_checkpoint_state(saved_ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    # saver.restore(sess, './checkpoint/resnet.model-30000')

    train_summary_writer = tf.summary.FileWriter(saved_summary_train_path, sess.graph)
    test_summary_writer = tf.summary.FileWriter(saved_summary_test_path, sess.graph)


    for i in range(0, MAX_STEPS + 1):
        train_img_data_batch, train_lables_batch = train_data.next_batch(BATCH_SIZE, 'train')
        test_img_data_batch, test_labels_batch = test_data.next_batch(BATCH_SIZE)

        train_summary, _ = sess.run([merged, optimizer], feed_dict={x: train_img_data_batch, y: train_lables_batch, keep_prob: KEEP_PROB})
        train_summary_writer.add_summary(train_summary, i)
        test_summary = sess.run(merged, feed_dict={x: test_img_data_batch, y: test_labels_batch, keep_prob: 1.0})
        test_summary_writer.add_summary(test_summary, i)


        train_accuracy, train_loss_val_all, train_loss_val = sess.run([accuracy, loss_all, loss],
                                                                      feed_dict={x: train_img_data_batch, y: train_lables_batch, keep_prob: 1.0})
        test_accuracy, test_loss_val_all, test_loss_val = sess.run([accuracy, loss_all, loss],
                                                                   feed_dict={x: test_img_data_batch, y: test_labels_batch, keep_prob: 1.0})

        if i % 50 == 0:
            learning_rate = sess.run(lr)
            print(
                "train step: %d, learning rate: %f, train loss all: %f, train loss: %f, test loss all: %f, test loss: %f" % (
                    i, learning_rate, train_loss_val_all, train_loss_val, test_loss_val_all, test_loss_val))

        if i % 200 == 0:
            learning_rate = sess.run(lr)
            print(
                "train step: %d, learning rate: %f, train loss all: %f, train loss: %f, test loss all: %f, test loss: %f, train accuracy: %f, test accuracy: %f" % (
                i, learning_rate, train_loss_val_all, train_loss_val, test_loss_val_all, test_loss_val, train_accuracy, test_accuracy))

        if i % 4000 == 0:
            saver.save(sess, os.path.join(saved_ckpt_path, 'resnet.model'), global_step=i)

        if i != 0:
            if i % 40000 == 0 or i % 60000 == 0:
                sess.run(tf.assign(lr, 0.1 * lr))
