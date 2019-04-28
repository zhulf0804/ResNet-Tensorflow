# coding=utf-8
from __future__ import print_function
from __future__ import division

import cPickle
import os
import numpy as np
from sklearn.utils import shuffle
#import tensorflow as tf
import cv2
import random
import math

DATA_DIR = './cifar-10-batches-py'
TRAIN_FILES = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
TEST_FILE = 'test_batch'

cmap = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

HEIGHT = 32
WIDTH = 32

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def flip_random_left_right(image):
    '''
    :param image: [height, width, channel]
    :return:
    '''
    flag = random.randint(0, 1)
    if flag:
        return cv2.flip(image, 1)
    return image

def flip_batch_random_left_right(batch_image):
    '''
    data augmentation: random flip horizontally
    :param batch_image: [batch, height, width, channel]
    :return:
    '''
    for i in range(len(batch_image)):
        batch_image[i, ...] = flip_random_left_right(batch_image[i, ...])

    return batch_image

def random_brightness(image, max_delta):
    '''

    :param image: [height, width, channel]
    :param max_delta: [-max_delta, max_delta]
    :return:
    '''
    #flag = random.randint(0, 1)
    flag = 1
    if flag:
        delta = random.randint(-max_delta, max_delta)
        return image + delta
    return image

def random_batch_brightness(batch_image, max_delta=32):
    '''
    data augmentation: random brightness
    :param batch_image: [batch, height, width, channel]
    :return:
    '''
    for i in range(len(batch_image)):
        batch_image[i, ...] = random_brightness(batch_image[i, ...], max_delta)

    return batch_image

def random_contrast(image, lower, upper):
    '''
    random contrast factor from [lower, upper]
    :param image: [height, width, channel]
    :param lower:
    :param upper:
    :return: (x - mean) * contrast_factor + mean
    '''
    mmean = np.mean(image)
    contrast_factor = random.uniform(lower, upper)
    #flag = random.randint(0, 1)
    flag = 1
    if flag:
        return (image - mmean) * contrast_factor + mmean
    return image

def random_batch_contrast(batch_image, lower=0.8, upper=1.2):
    '''

    :param batch_image: [batch, height, width, channel]
    :param lower:
    :param upper:
    :return:
    '''
    for i in range(len(batch_image)):
        batch_image[i, ...] = random_contrast(batch_image[i, ...], lower, upper)

    return batch_image

def image_pad_crop(image):

    #image = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    #image = np.pad(image, ((4, 4), (4, 4), (0, 0)), 'constant')

    y = random.randint(0, 8)
    x = random.randint(0, 8)
    #print(x, y)
    cropped_image = image[x:x+HEIGHT, y:y+WIDTH, :]

    return cropped_image

def image_batch_pad_crop(batch_image):

    cropped_batch = np.zeros(len(batch_image) * HEIGHT * WIDTH * 3).reshape(
        len(batch_image), HEIGHT, WIDTH, 3)

    for i in range(len(batch_image)):
        cropped_batch[i, ...] = image_pad_crop(batch_image[i, ...])

    return cropped_batch

def image_standardization(image):
    '''
    (image - mean) / adjusted_stddev, adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))
    :param image: [height, width, channel]
    :return:
    '''
    mmean = np.mean(image)
    stddev = np.std(image)
    num_elements = np.size(image)
    adjusted_stddev = np.maximum(stddev, 1.0/math.sqrt(num_elements))

    return (image - mmean) / adjusted_stddev

def image_standardization_batch(batch_image):


    for i in range(len(batch_image)):
        batch_image[i, ...] = image_standardization(batch_image[i, ...])

    return batch_image

def read_train_files():
    image_data = np.array([[]], dtype=np.uint8)
    labels = []
    for i in range(len(TRAIN_FILES)):
        train_file = TRAIN_FILES[i]
        train_file_path = os.path.join(DATA_DIR, train_file)
        train_data = unpickle(train_file_path)
        image_data_part = train_data['data']
        labels_part = train_data['labels']
        #print(image_data_part.shape)
        if i == 0:
            image_data = image_data_part
        else:
            image_data = np.concatenate((image_data, image_data_part), axis=0)
        labels += labels_part
    

    labels = np.array(labels, dtype=np.int32)
    #print(labels.dtype)

    image_data = np.reshape(image_data, [50000, 3, 32, 32])
    image_data = image_data.transpose(0, 2, 3, 1)

    return image_data, labels


def read_test_files():
    train_file_path = os.path.join(DATA_DIR, TEST_FILE)
    test_data = unpickle(train_file_path)
    image_data = test_data['data']
    labels = test_data['labels']
    labels = np.array(labels, dtype=np.int32)

    image_data = np.reshape(image_data, [10000, 3, 32, 32])
    image_data = image_data.transpose(0, 2, 3, 1)

    #cv2.imwrite('a.png', image_data[1000])
    return image_data, labels

class Dataset(object):

    def __init__(self, image_data, labels):
        self._num_examples = len(labels)
        self._image_data = image_data
        self._labels = labels
        self._epochs_done = 0
        self._index_in_epoch = 0
        self._flag = 0

    def next_batch(self, batch_size, type='test'):
        '''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

            self._image_data, self._labels = shuffle(self._image_data, self._labels)

        end = self._index_in_epoch

        image_data_val = self._image_data[start:end]
        '''

        if type == 'train':

            offset = np.random.choice(50000 - batch_size, 1)[0]

            image_data_val = self._image_data[offset:offset+batch_size, ...]

            # for augmentation
            image_data_val = image_batch_pad_crop(image_data_val)
            image_data_val = flip_batch_random_left_right(image_data_val)

            image_data_val = image_data_val.astype(np.float32) # uint8 -> float32

            #image_data_val = random_batch_brightness(image_data_val, max_delta=63)
            #image_data_val = random_batch_contrast(image_data_val, lower=0.8, upper=1.2)


        else:
            offset = np.random.choice(10000 - batch_size, 1)[0]
            image_data_val = self._image_data[offset:offset + batch_size, ...]
            image_data_val = image_data_val.astype(np.float32)  # uint8 -> float32
        # image_standardization

        image_data_val = image_standardization_batch(image_data_val)


        return image_data_val, self._labels[offset:offset+batch_size]


def read_train_data():
    train_image_data, train_labels = read_train_files()

    padding_size = 4
    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    train_image_data = np.pad(train_image_data, pad_width=pad_width, mode='constant', constant_values=0)

    train_image_data, train_labels = shuffle(train_image_data, train_labels)
    train_data = Dataset(train_image_data, train_labels)
    return train_data

def read_test_data():
    test_image_data, test_labels = read_test_files()
    test_image_data, test_labels = shuffle(test_image_data, test_labels)
    test_data = Dataset(test_image_data, test_labels)
    return test_data

if __name__ == '__main__':

    train_image_data, train_labels = read_train_files()
    padding_size = 4
    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    train_image_data = np.pad(train_image_data, pad_width=pad_width, mode='constant', constant_values=0)
    train_image_data, train_labels = shuffle(train_image_data, train_labels)
    train_data = Dataset(train_image_data, train_labels)

    test_image_data, test_labels = read_test_files()
    test_image_data, test_labels = shuffle(test_image_data, test_labels)
    test_data = Dataset(test_image_data, test_labels)

    train_img_data_batch, train_lables_batch = train_data.next_batch(4, 'train')

    test_img_data_batch, test_labels_batch = test_data.next_batch(4)

    for i in range(4):
        cv2.imwrite('./test/train_%d_%s.png' %(i, cmap[train_lables_batch[i]]), train_img_data_batch[i, ...])
        cv2.imwrite('./test/test_%d_%s.png' %(i, cmap[test_labels_batch[i]]), test_img_data_batch[i, ...])


    #print(train_img_data_tensor)
    #print(test_img_data_tensor)



