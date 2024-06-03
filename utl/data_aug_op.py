import random

import cv2
import numpy as np


def random_flip_img(img, horizontal_chance=0, vertical_chance=0):
    """

    Parameters
    ----------
    img:  np.ndarray containing an image
    horizontal_chance: the probability of flipping horizontally the image
    vertical_chance: the probability of flipping vertically the image

    Returns
    -------
    img: flipped image
    """
    flip_horizontal = False
    if random.random() < horizontal_chance:
        flip_horizontal = True

    flip_vertical = False
    if random.random() < vertical_chance:
        flip_vertical = True

    if not flip_horizontal and not flip_vertical:
        return img

    flip_val = 1
    if flip_vertical:
        flip_val = -1 if flip_horizontal else 0

    if not isinstance(img, list):
        res = cv2.flip(img, flip_val)
    else:
        res = []
        for img_item in img:
            img_flip = cv2.flip(img_item, flip_val)
            res.append(img_flip)
    return res


def random_rotate_img(images):
    """

    Parameters
    ----------
    images: np.ndarray of an image

    Returns
    -------
    img_inst: a randomly rotated image
    """
    rand_roat = np.random.randint(4, size=1)
    angle = 90 * rand_roat
    center = (images.shape[0] / 2, images.shape[1] / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle[0], scale=1.0)

    img_inst = cv2.warpAffine(images, rot_matrix, dsize=images.shape[:2], borderMode=cv2.BORDER_CONSTANT)

    return img_inst
