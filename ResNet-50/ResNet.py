"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     7/1/2017
Comments: ResNet with 50 convolutional layer implemented on the famous MNIST data.

The general structure of the network is similar to the one with 50 layer used in the original
paper: "Deep Residual Learning for Image Recognition" (see Table 1); of course with changes 
in the network parameters such as number of filters in each of the convolutional layers, 
kernel sizes, etc. to make it compatible with the MNIST images.
**********************************************************************************
"""

import tensorflow as tf
from ops import *
from utils import *
from network import create_network


class ResNet:
    # Class properties
    __network = None  # Graph for ResNet
    __train_op = None  # Operation used to optimize loss function
    __loss = None  # Loss function to be optimized, which is based on predictions
    __accuracy = None  # Classification accuracy for all conditions
    __probs = None  # Prediction probability matrix of shape [batch_size, numClasses]

    def __init__(self, numClasses, imgSize, imgChannel):
        self.imgSize = imgSize
        self.imgChannel = imgChannel
        self.numClasses = numClasses
        self.h = 100  # number of hidden units of the fully-connected layer
        self.lmbda = 5e-04  # weight decay
        self.init_lr = 0.001  # initial learning rate
        self.x, self.y, self.keep_prob = self.create_placeholders()
        self.is_train = tf.Variable(True, trainable=False, dtype=tf.bool)

    def create_placeholders(self):
        with tf.name_scope('Input'):
            x = tf.placeholder(tf.float32, shape=(None,
                                                  self.imgSize,
                                                  self.imgSize,
                                                  self.imgChannel), name='x-input')
            y = tf.placeholder(tf.float32, shape=(None, self.numClasses), name='y-input')
            keep_prob = tf.placeholder(tf.float32)
        return x, y, keep_prob

    def inference(self):
        if self.__network:
            return self
        # Building network...
        with tf.variable_scope('ResNet'):
            net = create_network(self.x, self.h, self.keep_prob, self.numClasses, self.is_train)
        self.__network = net
        return self

    def pred_func(self):
        if self.__probs:
            return self
        self.__probs = tf.nn.softmax(self.__network)
        return self

    def accuracy_func(self):
        if self.__accuracy:
            return self
        with tf.name_scope('Accuracy'):
            self.__accuracy = accuracy_generator(self.y, self.__network)
        return self

    def loss_func(self):
        if self.__loss:
            return self
        with tf.name_scope('Loss'):
            with tf.name_scope('cross_entropy'):
                cross_entropy = cross_entropy_loss(self.y, self.__network)
                tf.summary.scalar('cross_entropy', cross_entropy)
            with tf.name_scope('l2_loss'):
                l2_loss = tf.reduce_sum(
                    self.lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('reg_weights')]))
                tf.summary.scalar('l2_loss', l2_loss)
            with tf.name_scope('total'):
                self.__loss = cross_entropy + l2_loss
        return self

    def train_func(self):
        if self.__train_op:
            return self
        with tf.name_scope('Train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.init_lr)
            self.__train_op = optimizer.minimize(self.__loss)
        return self

    @property
    def probs(self):
        return self.__probs

    @property
    def network(self):
        return self.__network

    @property
    def train_op(self):
        return self.__train_op

    @property
    def loss(self):
        return self.__loss

    @property
    def accuracy(self):
        return self.__accuracy
