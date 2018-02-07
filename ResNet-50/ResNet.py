"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     7/1/2017
Comments: ResNet-50 implementation for Chest X-ray data set.

The general structure of the network is similar to the one with 50 layer used in the original
paper: "Deep Residual Learning for Image Recognition" (see Table 1); of course with changes 
in the network parameters such as number of filters in each of the convolutional layers, 
kernel sizes, etc. to make it compatible with our images.
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
    __logits = None  # logits of shape [batch_size, numClasses]
    __preds = None  # Prediction probability matrix of shape [batch_size, numClasses]
    __features = None
    __cls_act_map = None
    __vol_features = None

    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=(None, args.img_h, args.img_w, args.n_ch), name='x-input')
        self.y = tf.placeholder(tf.float32, shape=(None, args.n_cls), name='y-input')
        self.w_plus = tf.placeholder(tf.float32, shape=args.n_cls, name='w_plus')
        self.weighted_loss = tf.placeholder_with_default(True, shape=(), name="mask_with_labels")
        self.is_train = True

    def inference(self):
        if self.__network:
            return self
        # Building network...
        with tf.variable_scope('ResNet'):
            net, net_flatten, vol_feat, cls_act_map = create_network(self.x, args.n_cls, self.is_train)
        self.__network = net
        self.__features = net_flatten
        self.__vol_features = vol_feat
        self.__cls_act_map = cls_act_map
        # cls_act_map: [64, 14]
        return self

    def pred_func(self):
        if self.__preds:
            return self
        self.__logits = tf.nn.sigmoid(self.__network)
        self.__preds = tf.round(tf.nn.sigmoid(self.__network))
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
                cross_entropy = cross_entropy_loss(self.y, self.__network,
                                                   self.w_plus, weighted_loss=self.weighted_loss)
                tf.summary.scalar('cross_entropy', cross_entropy)
            with tf.name_scope('l2_loss'):
                l2_loss = tf.reduce_sum(
                    args.lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('reg_weights')]))
                tf.summary.scalar('l2_loss', l2_loss)
            with tf.name_scope('total'):
                self.__loss = cross_entropy + l2_loss
        return self

    def train_func(self):
        if self.__train_op:
            return self
        with tf.name_scope('Train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=args.init_lr)
            self.__train_op = optimizer.minimize(self.__loss)
        return self

    @property
    def get_cls_act_map(self):
        return self.__cls_act_map

    @property
    def get_features(self):
        return self.__features

    @property
    def get_vol_features(self):
        return self.__vol_features

    @property
    def get_logits(self):
        return self.__logits

    @property
    def prediction(self):
        return self.__preds

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
