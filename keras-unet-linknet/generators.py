import numpy as np
import os
import cv2


def train_generator(batch_size, shape, train_df, preprocessing_fn, aug):
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
                image = aug.augment_image(image)
                seq_2_det = aug.to_deterministic()
                image = aug.augment_image(image)

                mask_lungs = seq_2_det.augment_image(mask_lungs)
                contour_lungs = seq_2_det.augment_image(contour_lungs)
                mask_heart = seq_2_det.augment_image(mask_heart)
                contour_heart = seq_2_det.augment_image(contour_heart)
                mask_clavicles = seq_2_det.augment_image(mask_clavicles)
                contour_clavicles = seq_2_det.augment_image(contour_clavicles)

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
