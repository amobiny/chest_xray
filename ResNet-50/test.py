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
from config import args
import h5py


def test():
    # load the test data
    X_test, Y_test = load_data(args.img_w, args.n_cls, args.n_ch, mode='test')
    print('Test set', X_test.shape, Y_test.shape)
    # load the model
    model = ResNet()
    model.inference().accuracy_func().loss_func().train_func()

    saver = tf.train.Saver()
    save_path = os.path.join(args.load_dir, '20180202-182812/' + 'model_54')
    w_plus = (Y_test.shape[0] - np.sum(Y_test, axis=0)) / (np.sum(Y_test, axis=0))

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        print("Model restored.")
        # features = (X_test, Y_test, args.batch_size, sess, model, w_plus, after_pooling=True)
        # vol_features = compute_features(X_test, Y_test, args.batch_size, sess, model, w_plus, after_pooling=False)
        # cls_act_map = get_act_map(X_test, Y_test, args.batch_size, sess, model, w_plus)

        features, vol_features, cls_act_map = get_all(X_test, Y_test, args.val_batch_size, sess, model, w_plus)

    h5f = h5py.File('features_no_normal_fix_weight.h5', 'w')
    h5f.create_dataset('features', data=features)
    h5f.create_dataset('X_test', data=X_test[:features.shape[0]])
    h5f.create_dataset('Y_test', data=Y_test[:features.shape[0]])
    h5f.close()

    h5f = h5py.File('vol_features_no_normal_fix_weight.h5', 'w')
    h5f.create_dataset('features', data=vol_features)
    h5f.create_dataset('X_test', data=X_test[:features.shape[0]])
    h5f.create_dataset('Y_test', data=Y_test[:features.shape[0]])
    h5f.close()

    h5f = h5py.File('cls_act_map_no_normal_fix_weight.h5', 'w')
    h5f.create_dataset('cls_act_map', data=cls_act_map)
    h5f.close()


if __name__ == '__main__':
    test()
