"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     7/1/2017
Comments: Includes functions for defining the ResNet layers
**********************************************************************************
"""

import tensorflow as tf
from tflearn.layers.normalization import batch_normalization


def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W_' + name, dtype=tf.float32,
                           shape=shape, initializer=initer)


def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initial bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name, dtype=tf.float32,
                           initializer=initial)


def fc_layer(bottom, out_dim, layer_name, is_train=True, batch_norm=False, add_reg=False, use_relu=True):
    """
    Creates a fully-connected layer
    :param bottom: input from previous layer
    :param out_dim: number of hidden units in the fully-connected layer
    :param layer_name: layer name
    :param is_train: boolean to differentiate train and test (useful when applying batch normalization)
    :param batch_norm: boolean to add the batch normalization layer (or not)
    :param add_reg: boolean to add norm-2 regularization (or not)
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    in_dim = bottom.get_shape()[1]
    with tf.variable_scope(layer_name):
        weights = weight_variable(layer_name, shape=[in_dim, out_dim])
        tf.summary.histogram('W', weights)
        biases = bias_variable(layer_name, [out_dim])
        layer = tf.matmul(bottom, weights)
        if batch_norm:
            layer = batch_norm_wrapper(layer, is_train)
        layer += biases
        if use_relu:
            layer = tf.nn.relu(layer)
        if add_reg:
            tf.add_to_collection('weights', weights)
    return layer


def conv_2d(inputs, filter_size, stride, num_filters, layer_name,
            is_train=True, batch_norm=False, add_reg=False, use_relu=True):
    """
    Create a 2D convolution layer
    :param inputs: input array
    :param filter_size: size of the filter
    :param stride: filter stride
    :param num_filters: number of filters (or output feature maps)
    :param layer_name: layer name
    :param is_train: boolean to differentiate train and test (useful when applying batch normalization)
    :param batch_norm: boolean to add the batch normalization layer (or not)
    :param add_reg: boolean to add norm-2 regularization (or not)
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    num_in_channel = inputs.get_shape().as_list()[-1]
    with tf.variable_scope(layer_name):
        shape = [filter_size, filter_size, num_in_channel, num_filters]
        weights = weight_variable(layer_name, shape=shape)
        tf.summary.histogram('W', weights)
        biases = bias_variable(layer_name, [num_filters])
        layer = tf.nn.conv2d(input=inputs,
                             filter=weights,
                             strides=[1, stride, stride, 1],
                             padding="SAME")
        print('{}: {}'.format(layer_name, layer.get_shape()))
        if batch_norm:
            layer = batch_norm_wrapper(layer, is_train)
        layer += biases
        if use_relu:
            layer = tf.nn.relu(layer)
        if add_reg:
            tf.add_to_collection('weights', weights)
    return layer


def flatten_layer(layer):
    """
    Flattens the output of the convolutional layer to be fed into fully-connected layer
    :param layer: input array
    :return: flattened array
    """
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat


def max_pool(x, ksize, stride, name):
    """
    Create a max pooling layer
    :param x: input to max-pooling layer
    :param ksize: size of the max-pooling filter
    :param stride: stride of the max-pooling filter
    :param name: layer name
    :return: The output array
    """
    maxpool = tf.nn.max_pool(x,
                             ksize=[1, ksize, ksize, 1],
                             strides=[1, stride, stride, 1],
                             padding="SAME",
                             name=name)
    print('{}: {}'.format(maxpool.name, maxpool.get_shape()))
    return maxpool


def avg_pool(x, ksize, stride, name):
    """Create an average pooling layer."""
    return tf.nn.avg_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding="VALID",
                          name=name)


def dropout(x, keep_prob):
    """
    Create a dropout layer
    :param x: input to dropout layer
    :param keep_prob: dropout rate (e.g.: 0.5 means keeping 50% of the units)
    :return: the output array
    """
    return tf.nn.dropout(x, keep_prob)


def batch_norm_wrapper(inputs, is_training, decay=0.999, epsilon=1e-3):
    """
    creates a batch normalization layer
    :param inputs: input array
    :param is_training: boolean for differentiating train and test
    :param decay:
    :param epsilon:
    :return: normalized input
    """
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if len(inputs.get_shape().as_list()) == 4:  # For convolutional layers
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:  # For fully-connected layers
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)


def bottleneck_block(x, is_train, block_name,
                     s1, k1, nf1, name1,
                     s2, k2, nf2, name2,
                     s3, k3, nf3, name3,
                     s4, k4, name4, first_block=False):
    with tf.variable_scope(block_name):
        # Convolutional Layer 1
        layer_conv1 = conv_2d(x,
                              layer_name=name1,
                              stride=s1,
                              filter_size=k1,
                              num_filters=nf1,
                              is_train=is_train,
                              batch_norm=True,
                              use_relu=True)

        # Convolutional Layer 2
        layer_conv2 = conv_2d(layer_conv1,
                              layer_name=name2,
                              stride=s2,
                              filter_size=k2,
                              num_filters=nf2,
                              is_train=is_train,
                              batch_norm=True,
                              use_relu=True)

        # Convolutional Layer 3
        layer_conv3 = conv_2d(layer_conv2,
                              layer_name=name3,
                              stride=s3,
                              filter_size=k3,
                              num_filters=nf3,
                              is_train=is_train,
                              batch_norm=True,
                              use_relu=False)

        if first_block:
            shortcut = conv_2d(x,
                               layer_name=name4,
                               stride=s4,
                               filter_size=k4,
                               num_filters=nf3,
                               is_train=is_train,
                               batch_norm=True,
                               use_relu=False)
            assert (
                shortcut.get_shape().as_list() == layer_conv3.get_shape().as_list()), "Tensor sizes of the two branches are not matched!"
            res = shortcut + layer_conv3
        else:
            res = layer_conv3 + x
            assert (
                x.get_shape().as_list() == layer_conv3.get_shape().as_list()), "Tensor sizes of the two branches are not matched!"
    return tf.nn.relu(res)
