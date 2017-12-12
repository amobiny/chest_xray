import h5py
from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf

h5f = h5py.File('/data/Chest_Xray/data/chest256_train_801010.h5', 'r')
X_train = h5f['X_train'][:]
Y_train = h5f['Y_train'][:]
h5f.close()

tfrecords_filename = 'train.tfrecords'

original_pair = []
for i in range(X_train.shape[0]):
    original_pair.append((X_train[i], Y_train[i]))

height, width = original_pair[0][0].shape


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


writer = tf.python_io.TFRecordWriter(tfrecords_filename)
for img, label in original_pair:
    img_raw = img.tostring()
    label_raw = label.tostring()
    feature = {'height': _int64_feature(height),
               'width': _int64_feature(width),
               'image_raw': _bytes_feature(img_raw),
               'label_raw': _bytes_feature(label_raw)}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
writer.close()

# Let's check if the reconstructed images matchthe original images
# reconstructed_pair = []
# record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
# for string_record in record_iterator:
#     example = tf.train.Example()
#     example.ParseFromString(string_record)
#
#     height = int(example.features.feature['height'].int64_list.value[0])
#     width = int(example.features.feature['width'].int64_list.value[0])
#     img_string = (example.features.feature['image_raw'].bytes_list.value[0])
#     label_string = (example.features.feature['label_raw'].bytes_list.value[0])
#
#     img_1d = np.fromstring(img_string, dtype=np.float64)
#     reconstructed_img = img_1d.reshape((-1, height, width))
#     lbl_1d = np.fromstring(label_string, dtype=np.float64)
#     reconstructed_lbl = lbl_1d.reshape((-1, 15))
#     reconstructed_pair.append((reconstructed_img, reconstructed_lbl))
#
# for orig_pair, recons_pair in zip(original_pair, reconstructed_pair):
#     img_pair_to_compare, label_pair_to_compare = zip(orig_pair, recons_pair)
#     print(np.allclose(*img_pair_to_compare))
#     print(np.allclose(*label_pair_to_compare))

print()
