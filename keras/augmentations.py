from imgaug import augmenters as iaa

seq_2 = iaa.Sequential([
    iaa.Fliplr(0.5),

    iaa.OneOf([
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            rotate=(-15, 15),
            shear=(-10, 10),
        ),
        iaa.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            shear=(-10, 10),
        )
    ])
], random_order=True)
