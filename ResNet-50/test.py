"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     7/1/2017
Comments: Run this file to test the best saved model
**********************************************************************************
"""

import tensorflow as tf
from utils import *
from ResNet import ResNet
import os

save_dir = './checkpoints/'


def test(image_size=28,
         num_classes=10,
         num_channels=1):
    # load the test data
    X_test, Y_test = load_data(image_size, num_classes, num_channels, mode='test')
    print('Test set', X_test.shape, Y_test.shape)
    # load the model
    model = ResNet(num_classes, image_size, num_channels)
    model.inference().accuracy_func().loss_func().train_func()

    saver = tf.train.Saver()
    save_path = os.path.join(save_dir, 'best_validation')

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        print("Model restored.")
        feed_dict_test = {model.x: X_test, model.y: Y_test, model.keep_prob: 1}
        acc_test, loss_test = sess.run([model.accuracy, model.loss], feed_dict=feed_dict_test)
        print("Test loss: {0:.2f}, Test accuracy: {1:.01%}"
              .format(loss_test, acc_test))


if __name__ == '__main__':
    test(image_size=28,
         num_classes=10,
         num_channels=1)