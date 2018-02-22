"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     7/1/2017
Comments: Python utility functions
**********************************************************************************
"""
import csv
import os
import h5py
import numpy as np
import random
import scipy
import scipy.ndimage
import tensorflow as tf
from config import args
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def load_data(image_size, num_classes, num_channels, mode='train', with_normal=False):
    dir_path_parent = os.path.dirname(os.getcwd())
    if with_normal:
        dir_path_train = dir_path_parent + '/data/chest256_train_801010.h5'
        dir_path_valid = dir_path_parent + '/data/chest256_val_801010.h5'
        dir_path_test = dir_path_parent + '/data/chest256_test_801010.h5'
    else:
        dir_path_train = dir_path_parent + '/data/chest256_train_801010_no_normal.h5'
        dir_path_valid = dir_path_parent + '/data/chest256_val_801010_no_normal.h5'
        dir_path_test = dir_path_parent + '/data/chest256_test_801010_no_normal.h5'
    if mode == 'train':
        h5f_train = h5py.File(dir_path_train, 'r')
        x_train = h5f_train['X_train'][:] / 255.
        y_train = h5f_train['Y_train'][:]
        h5f_train.close()
        h5f_valid = h5py.File(dir_path_valid, 'r')
        x_valid = h5f_valid['X_val'][:] / 255.
        y_valid = h5f_valid['Y_val'][:]
        h5f_valid.close()
        x_train, _ = reformat(x_train, y_train, image_size, num_channels, num_classes)
        x_valid, _ = reformat(x_valid, y_valid, image_size, num_channels, num_classes)
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        h5f_test = h5py.File(dir_path_test, 'r')
        x_test = h5f_test['X_test'][:] / 255.
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
    dataset = x.reshape((-1, img_size, img_size, num_ch)).astype(np.float32)
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


def accuracy_generator(labels_tensor, logits_tensor):
    """
     Calculates the classification accuracy.
    :param labels_tensor: Tensor of correct predictions of size [batch_size, numClasses]
    :param logits_tensor: Predicted scores (logits) by the model.
            It should have the same dimensions as labels_tensor
    :return: accuracy: average accuracy over the samples of the current batch for each condition
    :return: avg_accuracy: average accuracy over all conditions
    """
    predictions = tf.nn.sigmoid(logits_tensor, name='predictions')
    correct_pred = tf.equal(labels_tensor, tf.round(predictions))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), axis=0)
    return accuracy


def cross_entropy_loss(labels_tensor, logits_tensor, w_plus, weighted_loss=True):
    """
     Calculates the cross-entropy loss function for the given parameters.
    :param labels_tensor: Tensor of correct predictions of size [batch_size, numClasses]
    :param logits_tensor: Predicted scores (logits) by the model.
            It should have the same dimensions as labels_tensor
    :return: Cross-entropy loss value over the samples of the current batch
    """
    if weighted_loss:
        labels_series = tf.unstack(labels_tensor, axis=1)
        logits_series = tf.unstack(logits_tensor, axis=1)
        losses_list = [tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels, pos_weight=w_p)
                       for (logits, labels, w_p) in zip(logits_series, labels_series, tf.split(w_plus, args.n_cls))]
        diff = tf.stack(losses_list, axis=1)
    else:
        diff = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_tensor, labels=labels_tensor)
    loss = tf.reduce_mean(diff)
    return loss


def plot_precision_recall_curve(y_true, logits, epoch_num):
    conditions = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                  'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                  'Pleural_Thickening', 'Hernia', 'Normal']
    color = ['gold', 'darkkhaki', 'chartreuse', 'darkgreen', 'deepskyblue', 'navy',
             'darkorchid', 'violet', 'crimson', 'brown', 'chocolate', 'gray', 'darkorange', 'pink', 'black']
    fig = plt.figure()
    fig.set_size_inches(12, 12)
    ax1 = fig.add_subplot(1, 1, 1)

    for cond in range(args.n_cls):
        Precision, Recall, thresholds = precision_recall_curve(y_true[:, cond], logits[:, cond])
        plt.plot(Recall, Precision, lw=2, color=color[cond], label=conditions[cond])
        ax1.set_xlabel('Recall', size=18)
        ax1.set_ylabel('Precision', size=18)
        ax1.tick_params(labelsize=18)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=7, fancybox=True, shadow=True)
    fig.savefig(args.results_dir + '/pr_' + str(epoch_num) + '.png')


def precision_recall(y_true, y_pred):
    """
    Computes the precision and recall values for the positive class
    :param y_true: true labels
    :param y_pred: predicted labels
    """
    TP = FP = FN = TN = 0
    for i in range(len(y_pred)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
    precision = (TP * 100.0) / (TP + FP + 0.001)
    recall = (TP * 100.0) / (TP + FN)
    return precision, recall


def create_acc_loss_file(fields):
    fields.append('loss')
    fields.insert(0, 'epoch')
    with open(args.results_dir + '/valid_acc_loss.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


def create_precision_recall_file(fields):
    fields.remove('loss')
    with open(args.results_dir + '/valid_precision_recall.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    fields.remove('epoch')


def write_acc_loss_csv(m_acc, m_loss, epo):
    fields = m_acc.tolist()
    fields.append(m_loss)
    fields.insert(0, epo)
    with open(args.results_dir + '/valid_acc_loss.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


def write_precision_recall_csv(precision, recall, epo):
    fields_p = precision.tolist()
    fields_r = recall.tolist()
    fields_p.insert(0, epo)
    fields_r.insert(0, epo)
    with open(args.results_dir + '/valid_precision_recall.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields_p)
        writer.writerow(fields_r)
