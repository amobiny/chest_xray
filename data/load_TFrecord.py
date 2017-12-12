import tensorflow as tf
import skimage.io as io
import matplotlib.pyplot as plt

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
num_classes = 15
tfrecords_filename = 'train.tfrecords'


def read_and_decode(filename_queue):
    """
    Read a TFRecords file
    :param filename_queue: Read a TFRecords file name
    :return: batch images and associated labels
    """

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    feature = {'height': tf.FixedLenFeature([], tf.int64),
               'width': tf.FixedLenFeature([], tf.int64),
               'image_raw': tf.FixedLenFeature([], tf.string),
               'label_raw': tf.FixedLenFeature([], tf.string)}
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert the image data from string back to the numbers
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.reshape(tf.decode_raw(features['image_raw'], tf.float64), [IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    label = tf.reshape(tf.decode_raw(features['label_raw'], tf.float64), [num_classes])

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=10,
                                            capacity=300,
                                            num_threads=4,
                                            min_after_dequeue=110)
    return images, labels


filename_que = tf.train.string_input_producer([tfrecords_filename], num_epochs=10)
imgs, labls = read_and_decode(filename_que)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Let's read off 3 batches just for example
    for i in xrange(3):
        X_batch, Y_batch = sess.run([imgs, labls])
        print(X_batch[0, :, :, :].shape)

        # We selected the batch size of two
        # So we should get two image pairs in each batch
        # Let's make sure it is random
        # plt.figure()
        # plt.imshow(X_batch[0, :, :, :].reshape(256, 256))
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(X_batch[1, :, :, :].reshape(256, 256))
        # plt.show()

    coord.request_stop()
    coord.join(threads)
