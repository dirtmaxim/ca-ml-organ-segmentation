import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.losses import binary_crossentropy


def iou(y_true, y_pred):
    ious = []

    for im1, im2 in zip(y_true, y_pred):
        im1 = np.asarray(im1).astype(np.bool)
        im2 = np.asarray(im2).astype(np.bool)
        intersection = np.logical_and(im1, im2)
        union = np.logical_or(im1, im2)
        iou = np.sum(intersection) / np.sum(union)
        ious.append(iou)

    return np.mean(ious)


def hard_dice(y_true, y_pred):
    dices = []

    for im1, im2 in zip(y_true, y_pred):
        im1 = np.asarray(im1).astype(np.bool)
        im2 = np.asarray(im2).astype(np.bool)
        intersection = np.logical_and(im1, im2)
        d = np.float(2. * intersection.sum()) / (im1.sum() + im2.sum() + 1e-7)
        dices.append(d)

    return np.mean(dices)


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dc_loss(y_true, y_pred):
    b = binary_crossentropy(K.clip(y_true, K.epsilon(), 1.), K.clip(y_pred, K.epsilon(), 1.))
    d = dice_coef(y_true, y_pred)

    return 1 - K.log(d) + b


def focal_loss(y_true, y_pred, gamma=2, alpha=10):
    max_val = K.clip(-y_pred, 0, 1024)
    loss = y_pred - y_pred * y_true + max_val + K.log((K.exp(-max_val) + K.exp((-y_pred - max_val))))

    invprobs = tf.log_sigmoid(-y_pred * (y_true * 2.0 - 1.0))
    loss = K.exp((invprobs * gamma)) * loss

    return K.mean(loss)


def fd_loss(y_true, y_pred):
    f = focal_loss(K.clip(y_true, K.epsilon(), 1.), K.clip(y_pred, K.epsilon(), 1.))
    d = dice_coef(y_true, y_pred)
    return f - K.log(d)
