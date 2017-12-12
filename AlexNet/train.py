"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     6/1/2017
Comments: Run this file to train the Alexnet model and save the best trained model
**********************************************************************************
"""

from datetime import datetime
import time
import tensorflow as tf
from utils import *
from AlexNet import Alexnet
import os

now = datetime.now()
logs_path = "./graph/" + now.strftime("%Y%m%d-%H%M%S")
save_dir = './checkpoints/'
conditions = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']

def train(image_size=28,
          num_classes=10,
          num_channels=1,
          num_epochs=100,
          batch_size=128,
          display=100):
    # Loading the MNIST data
    X_train, Y_train, X_valid, Y_valid = load_data(image_size, num_classes, num_channels, mode='train')
    print('Training set', X_train.shape, Y_train.shape)
    print('Validation set', X_valid.shape, Y_valid.shape)

    # Creating the alexnet model
    model = Alexnet(num_classes, image_size, num_channels)
    model.inference().accuracy_func().loss_func().train_func()

    # Saving the best trained model (based on the validation accuracy)
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model_')
    best_validation_accuracy = 0

    loss_batch_all = np.array([])
    acc_batch_all = np.zeros((0, num_classes))
    sum_count = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Initialized")
        merged = tf.summary.merge_all()
        batch_writer = tf.summary.FileWriter(logs_path + '/batch/', sess.graph)
        valid_writer = tf.summary.FileWriter(logs_path + '/valid/')
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print('------------------------------Training, Epoch: {} ----------------------------------'.format(epoch + 1))
            X_train, Y_train = randomize(X_train, Y_train)
            step_count = int(len(X_train) / batch_size)
            for step in range(step_count):
                start = step * batch_size
                end = (step + 1) * batch_size
                X_batch, Y_batch = get_next_batch(X_train, Y_train, start, end)
                # X_batch = random_rotation_2d(X_batch, 20.0)
                feed_dict_batch = {model.x: X_batch, model.y: Y_batch, model.keep_prob: 0.5}

                _, acc_batch, loss_batch = sess.run([model.train_op,
                                                     model.accuracy,
                                                     model.loss],
                                                    feed_dict=feed_dict_batch)
                acc_batch_all = np.concatenate((acc_batch_all, acc_batch.reshape([1, 10])))
                loss_batch_all = np.append(loss_batch_all, loss_batch)

                if step > 0 and not (step % display):
                    mean_acc_cond = np.mean(acc_batch_all, axis=0)
                    mean_loss = np.mean(loss_batch_all)
                    print('step %i, training loss: %.2f, average training accuracy: %.2f' %
                          (step, mean_loss, np.mean(mean_acc_cond) * 100))
                    print(mean_acc_cond * 100)
                    summary_tr = tf.Summary(value=[tf.Summary.Value(tag='Mean_Accuracy',
                                                                    simple_value=np.mean(mean_acc_cond) * 100)])
                    batch_writer.add_summary(summary_tr, sum_count * display)
                    for cond in range(num_classes):
                        with tf.name_scope('Accuracy'):
                            summary_tr = tf.Summary(
                                value=[tf.Summary.Value(tag='Accuracy_'+conditions[cond],
                                                        simple_value=mean_acc_cond[cond] * 100)])
                            batch_writer.add_summary(summary_tr, sum_count * display)

                    summary_tr = tf.Summary(value=[tf.Summary.Value(tag='Mean_Loss', simple_value=mean_loss)])
                    batch_writer.add_summary(summary_tr, sum_count * display)
                    summary = sess.run(merged, feed_dict=feed_dict_batch)
                    batch_writer.add_summary(summary, sum_count * display)
                    sum_count += 1
                    loss_batch_all = np.array([])
                    acc_batch_all = np.zeros((0, num_classes))

            feed_dict_val = {model.x: X_valid, model.y: Y_valid, model.keep_prob: 1}
            acc_valid, loss_valid = sess.run([model.accuracy,
                                              model.loss],
                                             feed_dict=feed_dict_val)
            for cond in range(num_classes):
                summary_valid = tf.Summary(value=[tf.Summary.Value(tag='Accuracy_'+conditions[cond],
                                                                   simple_value=acc_valid[cond] * 100)])
                valid_writer.add_summary(summary_valid, sum_count * display)
            summary_valid = tf.Summary(value=[tf.Summary.Value(tag='Mean_Loss', simple_value=loss_valid * 100)])
            valid_writer.add_summary(summary_valid, sum_count * display)
            summary_valid = tf.Summary(value=[tf.Summary.Value(tag='Mean_Accuracy', simple_value=np.mean(acc_valid))])
            valid_writer.add_summary(summary_valid, sum_count * display)

            # save the model after each epoch
            saver.save(sess=sess, save_path=save_path + str(epoch))

            if np.mean(acc_valid) > best_validation_accuracy:
                # Update the best-known validation accuracy.
                best_validation_accuracy = np.mean(acc_valid)
                improved_str = '*'
            else:
                # An empty string to be printed below.
                # Shows that no improvement was found.
                improved_str = ''
            epoch_time = time.time() - epoch_start_time
            print('---------------------------------Validation------------------------------------------')
            print("Epoch {0}, run time: {1:.1f} seconds, loss: {2:.2f}, accuracy: {3:.01%}{4}"
                  .format(epoch + 1, epoch_time, loss_valid, np.mean(acc_valid), improved_str))


if __name__ == '__main__':
    train(image_size=28,
          num_classes=10,
          num_channels=1,
          num_epochs=10,
          batch_size=128,
          display=100)
