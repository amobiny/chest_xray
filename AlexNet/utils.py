"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     6/1/2017
Comments: Python utility functions
**********************************************************************************
"""

import os
import h5py
import numpy as np
import random
import scipy
from tensorflow.examples.tutorials.mnist import input_data


def load_data(image_size, num_classes, num_channels, mode='train'):
    dir_path_parent = os.path.dirname(os.getcwd())
    dir_path_train = dir_path_parent + '/data/chest256_train_801010.h5'
    dir_path_valid = dir_path_parent + '/data/chest256_val_801010.h5'
    dir_path_test = dir_path_parent + '/data/chest256_test_801010.h5'
    if mode == 'train':
        h5f_train = h5py.File(dir_path_train, 'r')
        x_train = h5f_train['X_train'][:]
        y_train = h5f_train['Y_train'][:]
        h5f_train.close()
        h5f_valid = h5py.File(dir_path_valid, 'r')
        x_valid = h5f_valid['X_val'][:]
        y_valid = h5f_valid['Y_val'][:]
        h5f_valid.close()
        x_train, _ = reformat(x_train, y_train, image_size, num_channels, num_classes)
        x_valid, _ = reformat(x_valid, y_valid, image_size, num_channels, num_classes)
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        h5f_test = h5py.File(dir_path_test, 'r')
        x_test = h5f_test['X_test'][:]
        y_test = h5f_test['Y_test'][:]
        h5f_test.close()
        x_test, _ = reformat(x_test, y_test, image_size, num_channels, num_classes)
    return x_test, y_test

def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def reformat(x, y, img_size, num_ch, num_class):
    """ Reformats the data to the format acceptable for the conv layers"""
    dataset = x.reshape(
        (-1, img_size, img_size, num_ch)).astype(np.float32)
    labels = (np.arange(num_class) == y[:, None]).astype(np.float32)
    return dataset, labels


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def random_rotation_2d(batch, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    max_angle: `float`. The maximum rotation angle.

    Returns:
    batch of rotated 2D images
    """
    size = batch.shape
    batch = np.squeeze(batch)
    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        if bool(random.getrandbits(1)):
            image = np.squeeze(batch[i])
            angle = random.uniform(-max_angle, max_angle)
            batch_rot[i] = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', reshape=False)
        else:
            batch_rot[i] = batch[i]
    return batch_rot.reshape(size)
