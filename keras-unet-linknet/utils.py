import os
import numpy as np
from keras import backend as K
from keras.losses import binary_crossentropy
import cv2
from matplotlib import pyplot as plt


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


def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def bc_dice_loss(y_true, y_pred):
    b = binary_crossentropy(K.clip(y_true, K.epsilon(), 1.), K.clip(y_pred, K.epsilon(), 1.))
    d = dice_coefficient(y_true, y_pred)

    return 1 - K.log(d) + b


def generator(batch_size, shape, train_df, seq, preprocessing_fn):
    size = train_df.shape[0]

    while True:
        position = 0

        while position < size:
            i = 0
            images = np.zeros((batch_size, shape, shape, 3), dtype=np.uint8)
            masks = np.zeros((batch_size, shape, shape, 6), dtype=np.uint8)

            while i < batch_size:
                if position >= size:
                    break

                row = train_df.iloc[position]
                image = cv2.imread("dataset_combined" + os.sep + row["jsrt_png_imgs"])
                image = cv2.resize(image, (shape, shape))

                mask_lungs = cv2.imread("dataset_combined" + os.sep + row["lungs_png_masks"], 0)
                mask_lungs = cv2.resize(mask_lungs, (shape, shape))

                contour_lungs = cv2.imread("dataset_combined" + os.sep + row["lungs_png_contours"], 0)
                contour_lungs = cv2.resize(contour_lungs, (shape, shape))

                mask_heart = cv2.imread("dataset_combined" + os.sep + row["heart_png_masks"], 0)
                mask_heart = cv2.resize(mask_heart, (shape, shape))

                contour_heart = cv2.imread("dataset_combined" + os.sep + row["heart_png_contours"], 0)
                contour_heart = cv2.resize(contour_heart, (shape, shape))

                mask_clavicles = cv2.imread("dataset_combined" + os.sep + row["clavicles_png_masks"], 0)
                mask_clavicles = cv2.resize(mask_clavicles, (shape, shape))

                contour_clavicles = cv2.imread("dataset_combined" + os.sep + row["clavicles_png_contours"], 0)
                contour_clavicles = cv2.resize(contour_clavicles, (shape, shape))

                image = preprocessing_fn(image)
                seq_det = seq.to_deterministic()
                image = seq_det.augment_image(image)

                mask_lungs = seq_det.augment_image(mask_lungs)
                contour_lungs = seq_det.augment_image(contour_lungs)
                mask_heart = seq_det.augment_image(mask_heart)
                contour_heart = seq_det.augment_image(contour_heart)
                mask_clavicles = seq_det.augment_image(mask_clavicles)
                contour_clavicles = seq_det.augment_image(contour_clavicles)

                mask_lungs[mask_lungs > 0] = 1
                contour_lungs[contour_lungs == 0] = 0
                mask_heart[mask_heart > 0] = 1
                contour_heart[contour_heart == 0] = 0
                mask_clavicles[mask_clavicles > 0] = 1
                contour_clavicles[contour_clavicles == 0] = 0
                mask = np.stack(
                    [mask_lungs, contour_lungs, mask_heart, contour_heart, mask_clavicles, contour_clavicles], axis=-1)

                images[i] = image
                masks[i] = mask
                i += 1
                position += 1

            masks.shape = (masks.shape[0], shape, shape, 6)

            yield images, masks


def show_augm(index, train_df, seq, preprocessing_fn):
    im, ma = next(generator(32, 512, train_df, seq, preprocessing_fn))
    fig, axarr = plt.subplots(1, 2, figsize=(10, 20))
    axarr[0].axis("off")
    axarr[1].axis("off")
    axarr[0].imshow(im[index])
    axarr[1].imshow(ma[index][:, :, 0])
    plt.show()


def load_val(val_df, shape, preprocessing_fn):
    val_images = np.zeros((val_df.shape[0], shape, shape, 3), dtype=np.uint8)
    val_masks = np.zeros((val_df.shape[0], shape, shape, 6), dtype=np.uint8)
    i = 0

    while i < val_df.shape[0]:
        row = val_df.iloc[i]

        image = cv2.imread("dataset_combined" + os.sep + row["jsrt_png_imgs"])
        image = cv2.resize(image, (shape, shape))

        mask_lungs = cv2.imread("dataset_combined" + os.sep + row["lungs_png_masks"], 0)
        mask_lungs = cv2.resize(mask_lungs, (shape, shape))

        contour_lungs = cv2.imread("dataset_combined" + os.sep + row["lungs_png_contours"], 0)
        contour_lungs = cv2.resize(contour_lungs, (shape, shape))

        mask_heart = cv2.imread("dataset_combined" + os.sep + row["heart_png_masks"], 0)
        mask_heart = cv2.resize(mask_heart, (shape, shape))

        contour_heart = cv2.imread("dataset_combined" + os.sep + row["heart_png_contours"], 0)
        contour_heart = cv2.resize(contour_heart, (shape, shape))

        mask_clavicles = cv2.imread("dataset_combined" + os.sep + row["clavicles_png_masks"], 0)
        mask_clavicles = cv2.resize(mask_clavicles, (shape, shape))

        contour_clavicles = cv2.imread("dataset_combined" + os.sep + row["clavicles_png_contours"], 0)
        contour_clavicles = cv2.resize(contour_clavicles, (shape, shape))

        image = preprocessing_fn(image)
        mask = np.stack([mask_lungs, contour_lungs, mask_heart, contour_heart, mask_clavicles, contour_clavicles],
                        axis=-1)

        val_images[i] = image
        val_masks[i] = mask
        i += 1

    val_images = np.array(val_images)
    val_masks = np.array(val_masks)
    val_masks.shape = (val_masks.shape[0], shape, shape, 6)

    return val_images, val_masks


def load_test(test_df, shape, preprocessing_fn):
    test_images = np.zeros((test_df.shape[0], shape, shape, 3), dtype=np.uint8)
    test_masks = np.zeros((test_df.shape[0], shape, shape, 6), dtype=np.uint8)
    i = 0

    while i < test_df.shape[0]:
        row = test_df.iloc[i]
        image = cv2.imread("dataset_combined" + os.sep + row["jsrt_png_imgs"])
        image = cv2.resize(image, (shape, shape))

        mask_lungs = cv2.imread("dataset_combined" + os.sep + row["lungs_png_masks"], 0)
        mask_lungs = cv2.resize(mask_lungs, (shape, shape))

        contour_lungs = cv2.imread("dataset_combined" + os.sep + row["lungs_png_contours"], 0)
        contour_lungs = cv2.resize(contour_lungs, (shape, shape))

        mask_heart = cv2.imread("dataset_combined" + os.sep + row["heart_png_masks"], 0)
        mask_heart = cv2.resize(mask_heart, (shape, shape))

        contour_heart = cv2.imread("dataset_combined" + os.sep + row["heart_png_contours"], 0)
        contour_heart = cv2.resize(contour_heart, (shape, shape))

        mask_clavicles = cv2.imread("dataset_combined" + os.sep + row["clavicles_png_masks"], 0)
        mask_clavicles = cv2.resize(mask_clavicles, (shape, shape))

        contour_clavicles = cv2.imread("dataset_combined" + os.sep + row["clavicles_png_contours"], 0)
        contour_clavicles = cv2.resize(contour_clavicles, (shape, shape))

        image = preprocessing_fn(image)
        mask = np.stack([mask_lungs, contour_lungs, mask_heart, contour_heart, mask_clavicles, contour_clavicles],
                        axis=-1)

        test_images[i] = image
        test_masks[i] = mask
        i += 1

    test_images = np.array(test_images)
    test_masks = np.array(test_masks)
    test_masks.shape = (test_masks.shape[0], shape, shape, 6)

    return test_images, test_masks
