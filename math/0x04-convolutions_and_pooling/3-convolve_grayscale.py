#!/usr/bin/env python3
"""
Custom padding convolution
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images
    Args:
        - images is a numpy.ndarray with shape (m, h, w) containing multiple
          grayscale images:
            :m: is the number of images
            :h: is the height in pixels of the images
            :w: is the width in pixels of the images
        - kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
          for the convolution
            :kh: is the height of the kernel
            :kw: is the width of the kernel
        - padding is a tuple of (ph, pw)
            * if ‘same’, performs a same convolution
            * if ‘valid’, performs a valid convolution
            * if a tuple:
                :ph: is the padding for the height of the image
                :pw: is the padding for the width of the image
            * the image should be padded with 0’s
    Notes: You are only allowed to use two for loops; any other loops of any
           kind are not allowed
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride[0], stride[1]

    if padding == 'same':
        # Padding for the output
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    if padding == 'valid':
        ph, pw = 0, 0

    if isinstance(padding, tuple):
        # Padding for the output
        ph, pw = padding[0], padding[1]

    # Creating the pad of zeros around the output matrix
    pad_img = np.pad(images,
                     pad_width=((0, 0), (ph, ph), (pw, pw)),
                     mode='constant',
                     constant_values=0)

    # Output matrix height and width
    oh = int(np.floor(((h + (2 * ph) - kh) / sh) + 1))
    ow = int(np.floor(((w + (2 * pw) - kw) / sw) + 1))

    # Creating the output matrix with shape (m, h, w) as the inital input
    output = np.zeros((m, oh, ow))

    # Loop over every pixel in the output
    for x in range(ow):
        for y in range(oh):
            x0 = x * sw
            y0 = y * sh
            x1 = x0 + kw
            y1 = y0 + kh
            output[:, y, x] = np.sum(pad_img[:, y0:y1, x0:x1] * kernel,
                                     axis=(1, 2))

    return output
