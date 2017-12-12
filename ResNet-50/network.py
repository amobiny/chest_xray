"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     7/1/2017
Comments: Includes function which creates the ResNet with 50 layer.
        For more information regarding the structure of the network,
        please refer to Table 1 of the original paper:
        "Deep Residual Learning for Image Recognition".
	This code is prepared for the MNIST data. To use it with your own data,
	feel free to change the parameters such as kernel sizes, strides, etc.
**********************************************************************************
"""

from ops import conv_2d, bottleneck_block, avg_pool, flatten_layer, fc_layer, dropout
import tensorflow as tf


def create_network(X, h, keep_prob, numClasses, is_train):
    """
    Building the Residual Network with 50 layer
    :param X: input
    :param h: number of units in the fully connected layer
    :param keep_prob: dropout rate
    :param numClasses: number of classes
    :param is_train: to be used by batch normalization
    :return:
    """
    res1 = conv_2d(X,
                   layer_name='res1',
                   stride=2,
                   filter_size=4,
                   num_filters=32,
                   is_train=is_train,
                   batch_norm=True,
                   use_relu=True)

    # res1 = max_pool(res1, ksize=2, stride=2, name='res1_max_pool')
    print('---------------------')
    print('Res1')
    print(res1.get_shape())
    print('---------------------')
    # Res2
    with tf.variable_scope('Res2'):
        res2a = bottleneck_block(res1, is_train, block_name='res2a',
                                 s1=1, k1=1, nf1=32, name1='res2a_branch2a',
                                 s2=1, k2=3, nf2=32, name2='res2a_branch2b',
                                 s3=1, k3=1, nf3=64, name3='res2a_branch2c',
                                 s4=1, k4=1, name4='res2a_branch1', first_block=True)
        print('Res2a')
        print(res2a.get_shape())
        print('---------------------')
        res2b = bottleneck_block(res2a, is_train, block_name='res2b',
                                 s1=1, k1=1, nf1=32, name1='res2b_branch2a',
                                 s2=1, k2=3, nf2=32, name2='res2b_branch2b',
                                 s3=1, k3=1, nf3=64, name3='res2b_branch2c',
                                 s4=1, k4=1, name4='res2b_branch1', first_block=False)
        print('Res2b')
        print(res2b.get_shape())
        print('---------------------')
        res2c = bottleneck_block(res2b, is_train, block_name='res2c',
                                 s1=1, k1=1, nf1=32, name1='res2c_branch2a',
                                 s2=1, k2=3, nf2=32, name2='res2c_branch2b',
                                 s3=1, k3=1, nf3=64, name3='res2c_branch2c',
                                 s4=1, k4=1, name4='res2c_branch1', first_block=False)
        print('Res2c')
        print(res2c.get_shape())
        print('---------------------')

    # Res3
    with tf.variable_scope('Res3'):
        res3a = bottleneck_block(res2c, is_train, block_name='res3a',
                                 s1=2, k1=1, nf1=48, name1='res3a_branch2a',
                                 s2=1, k2=3, nf2=48, name2='res3a_branch2b',
                                 s3=1, k3=1, nf3=128, name3='res3a_branch2c',
                                 s4=2, k4=1, name4='res3a_branch1', first_block=True)
        print('Res3a')
        print(res3a.get_shape())
        print('---------------------')
        res3b = bottleneck_block(res3a, is_train, block_name='res3b',
                                 s1=1, k1=1, nf1=48, name1='res3b_branch2a',
                                 s2=1, k2=3, nf2=48, name2='res3b_branch2b',
                                 s3=1, k3=1, nf3=128, name3='res3b_branch2c',
                                 s4=1, k4=1, name4='res2b_branch1', first_block=False)
        print('Res3b')
        print(res3b.get_shape())
        print('---------------------')
        res3c = bottleneck_block(res3b, is_train, block_name='res3c',
                                 s1=1, k1=1, nf1=48, name1='res3c_branch2a',
                                 s2=1, k2=3, nf2=48, name2='res3c_branch2b',
                                 s3=1, k3=1, nf3=128, name3='res3c_branch2c',
                                 s4=1, k4=1, name4='res3c_branch1', first_block=False)
        print('Res3c')
        print(res3c.get_shape())
        print('---------------------')
        res3d = bottleneck_block(res3c, is_train, block_name='res3d',
                                 s1=1, k1=1, nf1=48, name1='res3d_branch2a',
                                 s2=1, k2=3, nf2=48, name2='res3d_branch2b',
                                 s3=1, k3=1, nf3=128, name3='res3d_branch2c',
                                 s4=1, k4=1, name4='res3d_branch1', first_block=False)
        print('Res3d')
        print(res3d.get_shape())
        print('---------------------')

    # Res4
    with tf.variable_scope('Res4'):
        res4a = bottleneck_block(res3d, is_train, block_name='res4a',
                                 s1=2, k1=1, nf1=64, name1='res4a_branch2a',
                                 s2=1, k2=3, nf2=64, name2='res4a_branch2b',
                                 s3=1, k3=1, nf3=256, name3='res4a_branch2c',
                                 s4=2, k4=1, name4='res4a_branch1', first_block=True)
        print('---------------------')
        print('Res4a')
        print(res4a.get_shape())
        print('---------------------')
        res4b = bottleneck_block(res4a, is_train, block_name='res4b',
                                 s1=1, k1=1, nf1=64, name1='res4b_branch2a',
                                 s2=1, k2=3, nf2=64, name2='res4b_branch2b',
                                 s3=1, k3=1, nf3=256, name3='res4b_branch2c',
                                 s4=1, k4=1, name4='res4b_branch1', first_block=False)
        print('Res4b')
        print(res4b.get_shape())
        print('---------------------')
        res4c = bottleneck_block(res4b, is_train, block_name='res4c',
                                 s1=1, k1=1, nf1=64, name1='res4c_branch2a',
                                 s2=1, k2=3, nf2=64, name2='res4c_branch2b',
                                 s3=1, k3=1, nf3=256, name3='res4c_branch2c',
                                 s4=1, k4=1, name4='res4c_branch1', first_block=False)
        print('Res4c')
        print(res4c.get_shape())
        print('---------------------')
        res4d = bottleneck_block(res4c, is_train, block_name='res4d',
                                 s1=1, k1=1, nf1=64, name1='res4d_branch2a',
                                 s2=1, k2=3, nf2=64, name2='res4d_branch2b',
                                 s3=1, k3=1, nf3=256, name3='res4d_branch2c',
                                 s4=1, k4=1, name4='res4d_branch1', first_block=False)
        print('Res4d')
        print(res4d.get_shape())
        print('---------------------')
        res4e = bottleneck_block(res4d, is_train, block_name='res4e',
                                 s1=1, k1=1, nf1=64, name1='res4e_branch2a',
                                 s2=1, k2=3, nf2=64, name2='res4e_branch2b',
                                 s3=1, k3=1, nf3=256, name3='res4e_branch2c',
                                 s4=1, k4=1, name4='res4e_branch1', first_block=False)
        print('Res4e')
        print(res4e.get_shape())
        print('---------------------')
        res4f = bottleneck_block(res4e, is_train, block_name='res4f',
                                 s1=1, k1=1, nf1=64, name1='res4f_branch2a',
                                 s2=1, k2=3, nf2=64, name2='res4f_branch2b',
                                 s3=1, k3=1, nf3=256, name3='res4f_branch2c',
                                 s4=1, k4=1, name4='res4f_branch1', first_block=False)
        print('Res4f')
        print(res4f.get_shape())
        print('---------------------')

    # Res5
    with tf.variable_scope('Res5'):
        res5a = bottleneck_block(res4f, is_train, block_name='res5a',
                                 s1=1, k1=1, nf1=128, name1='res5a_branch2a',
                                 s2=1, k2=3, nf2=128, name2='res5a_branch2b',
                                 s3=1, k3=1, nf3=512, name3='res5a_branch2c',
                                 s4=1, k4=1, name4='res5a_branch1', first_block=True)
        print('---------------------')
        print('Res5a')
        print(res5a.get_shape())
        print('---------------------')
        res5b = bottleneck_block(res5a, is_train, block_name='res5b',
                                 s1=1, k1=1, nf1=128, name1='res5b_branch2a',
                                 s2=1, k2=3, nf2=128, name2='res5b_branch2b',
                                 s3=1, k3=1, nf3=512, name3='res5b_branch2c',
                                 s4=1, k4=1, name4='res5b_branch1', first_block=False)
        print('Res5b')
        print(res5b.get_shape())
        print('---------------------')
        res5c = bottleneck_block(res5b, is_train, block_name='res5c',
                                 s1=1, k1=1, nf1=128, name1='res5c_branch2a',
                                 s2=1, k2=3, nf2=128, name2='res5c_branch2b',
                                 s3=1, k3=1, nf3=512, name3='res5c_branch2c',
                                 s4=1, k4=1, name4='res5c_branch1', first_block=False)
        print('Res5c')
        print(res5c.get_shape())

        res5c = avg_pool(res5c, ksize=4, stride=1, name='res5_avg_pool')
        print('---------------------')
        print('Res5c after AVG_POOL')
        print(res5c.get_shape())
        print('---------------------')

    net_flatten = flatten_layer(res5c)
    print('---------------------')
    print('Matrix dimension to the first FC layer')
    print(net_flatten.get_shape())
    print('---------------------')
    net = fc_layer(net_flatten, h, 'FC1', is_train=is_train, batch_norm=True, add_reg=True, use_relu=True)
    net = dropout(net, keep_prob)
    net = fc_layer(net, numClasses, 'FC2', is_train=is_train, batch_norm=True, add_reg=True, use_relu=False)
    return net
