from utils import randomize, get_next_batch
import numpy as np
from config import args


def compute_features(x, y, batch_size, sess, model, w_plus, after_pooling=True):
    """
    Feeds the data to the network and extracts the features from the last layers
    :param x: input images of size (#imgs, args.img_h, args.img_w, 1)
    :param y: corresponding labels of size (#imgs, args.n_cls)
    :param batch_size: batch size
    :param sess: Tensorflow session
    :param model: Full network model
    :param w_plus: weights for the positive class (to be used in the loss function)
    :param after_pooling: boolean.
    If True, computes the features after the final global average pooling layer (#imgs, 2048)
    Else, computes the features before the pooling, (#imgs, 8, 8, 2048)
    :return:
    """
    model.is_train = False
    x, y = randomize(x, y)
    step_count = int(len(x) / batch_size)
    if after_pooling:
        all_features = np.zeros((0, 2048))
        for step in range(step_count):
            start = step * batch_size
            end = (step + 1) * batch_size
            x_batch, y_batch = get_next_batch(x, y, start, end)
            feed_dict_batch = {model.x: x_batch, model.y: y_batch, model.w_plus: w_plus}
            features = sess.run(model.get_features, feed_dict=feed_dict_batch)
            all_features = np.concatenate((all_features, features), axis=0)
        return all_features
    else:
        all_features = np.zeros((0, 8, 8, 2048))
        for step in range(step_count):
            start = step * batch_size
            end = (step + 1) * batch_size
            x_batch, y_batch = get_next_batch(x, y, start, end)
            feed_dict_batch = {model.x: x_batch, model.y: y_batch, model.w_plus: w_plus}
            features = sess.run(model.get_vol_features, feed_dict=feed_dict_batch)
            all_features = np.concatenate((all_features, features), axis=0)
        return all_features


def get_act_map(x, y, batch_size, sess, model, w_plus):
    """
    Computes the class activation maps
    :param x: input images of size (#imgs, args.img_h, args.img_w, 1)
    :param y: corresponding labels of size (#imgs, args.n_cls)
    :param batch_size: batch size
    :param sess: Tensorflow session
    :param model: Full network model
    :param w_plus: weights for the positive class (to be used in the loss function)
    :return: array of class activation maps, (#imgs, 64, args.n_cls)
    """
    model.is_train = False
    x, y = randomize(x, y)
    step_count = int(len(x) / batch_size)
    all_features = np.zeros((0, 64, args.n_cls))
    for step in range(step_count):
        start = step * batch_size
        end = (step + 1) * batch_size
        x_batch, y_batch = get_next_batch(x, y, start, end)
        feed_dict_batch = {model.x: x_batch, model.y: y_batch, model.w_plus: w_plus}
        features = sess.run(model.get_cls_act_map, feed_dict=feed_dict_batch)
        all_features = np.concatenate((all_features, features), axis=0)
    return all_features


def get_all(x, y, batch_size, sess, model, w_plus):
    """
    Combination of the above functions; sends out all features and activation-maps.
    :param x: input images of size (#imgs, args.img_h, args.img_w, 1)
    :param y: corresponding labels of size (#imgs, args.n_cls)
    :param batch_size: batch size
    :param sess: Tensorflow session
    :param model: Full network model
    :param w_plus: weights for the positive class (to be used in the loss function)
    :return:
    """
    model.is_train = False
    x, y = randomize(x, y)
    step_count = int(len(x) / batch_size)
    all_features = np.zeros((0, 2048))
    all_maps = np.zeros((0, 64, args.n_cls))
    all_vol_features = np.zeros((0, 8, 8, 2048))
    for step in range(step_count):
        start = step * batch_size
        end = (step + 1) * batch_size
        x_batch, y_batch = get_next_batch(x, y, start, end)
        feed_dict_batch = {model.x: x_batch, model.y: y_batch, model.w_plus: w_plus}
        features, vol_features, act_maps = sess.run([model.get_features,
                                                     model.get_vol_features,
                                                     model.get_cls_act_map],
                                                    feed_dict=feed_dict_batch)
        all_features = np.concatenate((all_features, features), axis=0)
        all_vol_features = np.concatenate((all_vol_features, vol_features), axis=0)
        all_maps = np.concatenate((all_maps, act_maps), axis=0)

    return all_features, all_vol_features, all_maps
